"""Full MobDef-Bench benchmark construction pipeline.

Orchestrates data loading, tokenization, user splitting, anomaly generation,
validation, and serialization.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..core.episode import SemanticEpisode, SemanticTrajectory
from .concept_generator import (
    CONCEPT_BY_ID,
    CONCEPT_DEFS,
    SPLIT_SEEN,
    SPLIT_UNKNOWN,
    SPLIT_ZS_COMP,
    SPLIT_ZS_FAMILY,
    ConceptDef,
    ConceptGenerator,
)
from .interventions import InterventionEngine, NUM_PRIMITIVES
from .synthetic_data import SyntheticMobilityGenerator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DATASET_NAMES = ["numosim", "geolife", "porto", "foursquare_nyc", "foursquare_tokyo"]

# Kinematic plausibility thresholds
MAX_EPISODE_LENGTH = 128
MIN_EPISODE_LENGTH = 2
MAX_TRIP_LENGTH_CHANGE = 20.0


@dataclass
class BenchmarkSplit:
    """Container for one split (train / val / test) of the benchmark."""
    normal: List[SemanticTrajectory] = field(default_factory=list)
    anomalous: List[SemanticTrajectory] = field(default_factory=list)
    concept_ids: List[int] = field(default_factory=list)
    primitive_labels: List[Optional[List[List[int]]]] = field(default_factory=list)


@dataclass
class Benchmark:
    """Complete benchmark artifact."""
    dataset_name: str = ""
    train: BenchmarkSplit = field(default_factory=BenchmarkSplit)
    val: BenchmarkSplit = field(default_factory=BenchmarkSplit)
    test: BenchmarkSplit = field(default_factory=BenchmarkSplit)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MobDefBenchBuilder:
    """End-to-end benchmark construction for MobDef-Bench."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.engine = InterventionEngine(seed=seed)
        self.concept_gen = ConceptGenerator(self.engine)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        datasets: Optional[List[str]] = None,
    ) -> Dict[str, Benchmark]:
        """Run the full benchmark construction pipeline.

        Steps:
            1. Load raw data (or generate synthetic if unavailable).
            2. Tokenize trajectories into SemanticEpisode sequences.
            3. Split users: 70 % train / 15 % val / 15 % test.
            4. Generate normal trajectories per split.
            5. Apply interventions to create anomalies per concept.
            6. Create concept splits (seen / zs_comp / zs_family / unknown).
            7. Validate kinematic plausibility and balance.
            8. Save benchmark to *output_dir*.

        Returns:
            Dict mapping dataset name to its Benchmark object.
        """
        if datasets is None:
            datasets = list(DATASET_NAMES)

        benchmarks: Dict[str, Benchmark] = {}

        for ds_name in datasets:
            trajectories = self._load_or_generate(ds_name)
            trajectories = self._tokenize(trajectories)

            train_trajs, val_trajs, test_trajs = self._split_by_user(trajectories)

            benchmark = Benchmark(dataset_name=ds_name)

            # Normal trajectories go into each split
            benchmark.train.normal = train_trajs
            benchmark.val.normal = val_trajs
            benchmark.test.normal = test_trajs

            # Generate anomalies ------------------------------------------
            # A_seen: anomalies in train + val + test
            seen_concepts = [c for c in CONCEPT_DEFS if c.split == SPLIT_SEEN]
            # A_zs_comp: val + test only
            zs_comp_concepts = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_COMP]
            # A_zs_family: test only
            zs_family_concepts = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_FAMILY]
            # A_unknown: test only
            unknown_concepts = [c for c in CONCEPT_DEFS if c.split == SPLIT_UNKNOWN]

            # Train: seen concepts only
            self._inject_anomalies(benchmark.train, train_trajs, seen_concepts)

            # Val: seen + zs_comp
            self._inject_anomalies(benchmark.val, val_trajs, seen_concepts + zs_comp_concepts)

            # Test: all concepts
            self._inject_anomalies(
                benchmark.test, test_trajs,
                seen_concepts + zs_comp_concepts + zs_family_concepts + unknown_concepts,
            )

            # Validate
            issues = self.validate_benchmark(benchmark)
            benchmark.metadata = {
                "dataset": ds_name,
                "seed": self.seed,
                "n_train_normal": len(benchmark.train.normal),
                "n_train_anomalous": len(benchmark.train.anomalous),
                "n_val_normal": len(benchmark.val.normal),
                "n_val_anomalous": len(benchmark.val.anomalous),
                "n_test_normal": len(benchmark.test.normal),
                "n_test_anomalous": len(benchmark.test.anomalous),
                "validation_issues": issues,
                "statistics": self.get_statistics(benchmark),
            }

            # Save
            self._save_benchmark(benchmark)
            benchmarks[ds_name] = benchmark

        return benchmarks

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_or_generate(self, dataset_name: str) -> List[SemanticTrajectory]:
        """Load dataset from disk or fall back to synthetic generation."""
        ds_path = self.data_dir / dataset_name
        if ds_path.exists():
            return self._load_dataset(ds_path)
        # Fall back to synthetic data
        return self.generate_synthetic_data(n_users=200, trips_per_user=100)

    def _load_dataset(self, ds_path: Path) -> List[SemanticTrajectory]:
        """Load pre-processed trajectories from a directory.

        Expected format: one ``.pt`` file per user containing a list of
        serialised trajectory dicts, or a single ``trajectories.pt`` file.
        """
        single_file = ds_path / "trajectories.pt"
        if single_file.exists():
            data = torch.load(str(single_file), map_location="cpu", weights_only=False)
            return self._deserialize_trajectories(data)

        # Try per-user files
        pt_files = sorted(ds_path.glob("*.pt"))
        trajectories: List[SemanticTrajectory] = []
        for pf in pt_files:
            data = torch.load(str(pf), map_location="cpu", weights_only=False)
            trajectories.extend(self._deserialize_trajectories(data))
        return trajectories

    @staticmethod
    def _deserialize_trajectories(
        data: Any,
    ) -> List[SemanticTrajectory]:
        """Convert raw loaded data into SemanticTrajectory objects."""
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], SemanticTrajectory):
                return data
            if isinstance(data[0], dict):
                result: List[SemanticTrajectory] = []
                for d in data:
                    episodes = [
                        SemanticEpisode.from_list(ep) for ep in d.get("episodes", [])
                    ]
                    result.append(SemanticTrajectory(
                        episodes=episodes,
                        user_id=d.get("user_id", ""),
                        trip_id=d.get("trip_id", ""),
                        label=d.get("label", 0),
                        primitive_labels=d.get("primitive_labels"),
                    ))
                return result
        return []

    # ------------------------------------------------------------------
    # Tokenization (identity if already tokenised)
    # ------------------------------------------------------------------

    def _tokenize(
        self, trajectories: List[SemanticTrajectory],
    ) -> List[SemanticTrajectory]:
        """Ensure all trajectories are tokenised SemanticEpisode sequences.

        If trajectories are already in factorised form (as produced by the
        synthetic generator or a prior tokenisation step), this is a no-op
        that simply validates and clamps values.
        """
        for traj in trajectories:
            for ep in traj.episodes:
                ep.poi_role = ep.poi_role % SemanticEpisode.POI_ROLE_VOCAB
                ep.time_bin = ep.time_bin % SemanticEpisode.TIME_BIN_VOCAB
                ep.dwell_bin = min(
                    max(ep.dwell_bin, 0), SemanticEpisode.DWELL_BIN_VOCAB - 1
                )
                ep.transition_type = ep.transition_type % SemanticEpisode.TRANSITION_VOCAB
        return trajectories

    # ------------------------------------------------------------------
    # User splitting
    # ------------------------------------------------------------------

    def _split_by_user(
        self, trajectories: List[SemanticTrajectory],
    ) -> Tuple[List[SemanticTrajectory], List[SemanticTrajectory], List[SemanticTrajectory]]:
        """Split trajectories by user into train / val / test."""
        user_ids = sorted({t.user_id for t in trajectories})
        self.rng.shuffle(user_ids)

        n = len(user_ids)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_users = set(user_ids[:n_train])
        val_users = set(user_ids[n_train : n_train + n_val])
        test_users = set(user_ids[n_train + n_val :])

        train = [t for t in trajectories if t.user_id in train_users]
        val = [t for t in trajectories if t.user_id in val_users]
        test = [t for t in trajectories if t.user_id in test_users]

        return train, val, test

    # ------------------------------------------------------------------
    # Anomaly injection
    # ------------------------------------------------------------------

    def _inject_anomalies(
        self,
        split: BenchmarkSplit,
        normal_pool: List[SemanticTrajectory],
        concepts: List[ConceptDef],
    ) -> None:
        """Generate anomalous trajectories and add them to *split*."""
        results = self.concept_gen.generate_all_concepts(normal_pool, concepts)
        for anom_traj, cid, prim_labels in results:
            split.anomalous.append(anom_traj)
            split.concept_ids.append(cid)
            split.primitive_labels.append(prim_labels)

    # ------------------------------------------------------------------
    # Synthetic data generation
    # ------------------------------------------------------------------

    def generate_synthetic_data(
        self,
        n_users: int = 200,
        trips_per_user: int = 100,
    ) -> List[SemanticTrajectory]:
        """Generate synthetic mobility data for development / sanity checks."""
        gen = SyntheticMobilityGenerator(
            n_users=n_users,
            n_zones=50,
            n_roles=SemanticEpisode.POI_ROLE_VOCAB,
            seed=self.seed,
        )
        return gen.generate(trips_per_user=trips_per_user)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_benchmark(self, benchmark: Benchmark) -> List[str]:
        """Check plausibility constraints and balance.

        Returns a list of human-readable issue strings (empty if all OK).
        """
        issues: List[str] = []

        for split_name, split in [
            ("train", benchmark.train),
            ("val", benchmark.val),
            ("test", benchmark.test),
        ]:
            # Length checks
            for traj in split.normal + split.anomalous:
                if len(traj.episodes) > MAX_EPISODE_LENGTH:
                    issues.append(
                        f"{split_name}: trajectory {traj.trip_id} exceeds "
                        f"max length ({len(traj.episodes)} > {MAX_EPISODE_LENGTH})"
                    )
                if len(traj.episodes) < MIN_EPISODE_LENGTH:
                    issues.append(
                        f"{split_name}: trajectory {traj.trip_id} is too short "
                        f"({len(traj.episodes)} < {MIN_EPISODE_LENGTH})"
                    )

            # Kinematic plausibility: trip_length_change
            for traj in split.anomalous:
                for ep in traj.episodes:
                    if ep.trip_length_change > MAX_TRIP_LENGTH_CHANGE:
                        issues.append(
                            f"{split_name}: trajectory {traj.trip_id} has "
                            f"implausible trip_length_change={ep.trip_length_change:.1f}"
                        )
                        break  # one per trajectory

            # Balance: anomalous should be 5-50 % of normal
            n_norm = len(split.normal)
            n_anom = len(split.anomalous)
            if n_norm > 0 and n_anom > 0:
                ratio = n_anom / n_norm
                if ratio < 0.01:
                    issues.append(
                        f"{split_name}: too few anomalies "
                        f"({n_anom}/{n_norm} = {ratio:.3f})"
                    )
                if ratio > 0.60:
                    issues.append(
                        f"{split_name}: too many anomalies "
                        f"({n_anom}/{n_norm} = {ratio:.3f})"
                    )

            # Concept coverage
            if split_name == "test" and split.concept_ids:
                present = set(split.concept_ids)
                expected = {c.concept_id for c in CONCEPT_DEFS}
                missing = expected - present
                if missing:
                    issues.append(
                        f"test: missing concepts {sorted(missing)}"
                    )

        return issues

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self, benchmark: Benchmark) -> Dict[str, Any]:
        """Return a summary dict with counts and distributions."""
        stats: Dict[str, Any] = {"dataset": benchmark.dataset_name}

        for split_name, split in [
            ("train", benchmark.train),
            ("val", benchmark.val),
            ("test", benchmark.test),
        ]:
            n_normal = len(split.normal)
            n_anomalous = len(split.anomalous)

            ep_lengths_normal = [len(t) for t in split.normal]
            ep_lengths_anom = [len(t) for t in split.anomalous]

            concept_dist: Dict[int, int] = {}
            for cid in split.concept_ids:
                concept_dist[cid] = concept_dist.get(cid, 0) + 1

            # Primitive activation frequency across anomalous episodes
            prim_counts = [0] * NUM_PRIMITIVES
            total_episodes = 0
            for plabels in split.primitive_labels:
                if plabels is not None:
                    for row in plabels:
                        for d in range(NUM_PRIMITIVES):
                            prim_counts[d] += row[d]
                        total_episodes += 1

            prim_freq = (
                [c / total_episodes for c in prim_counts]
                if total_episodes > 0
                else prim_counts
            )

            stats[split_name] = {
                "n_normal": n_normal,
                "n_anomalous": n_anomalous,
                "mean_episode_length_normal": (
                    float(np.mean(ep_lengths_normal)) if ep_lengths_normal else 0.0
                ),
                "mean_episode_length_anomalous": (
                    float(np.mean(ep_lengths_anom)) if ep_lengths_anom else 0.0
                ),
                "concept_distribution": concept_dist,
                "primitive_activation_freq": prim_freq,
                "n_unique_users_normal": len({t.user_id for t in split.normal}),
                "n_unique_users_anomalous": len({t.user_id for t in split.anomalous}),
            }

        return stats

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _save_benchmark(self, benchmark: Benchmark) -> None:
        """Save benchmark splits as .pt files and metadata as JSON."""
        out_dir = self.output_dir / benchmark.dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split in [
            ("train", benchmark.train),
            ("val", benchmark.val),
            ("test", benchmark.test),
        ]:
            payload = {
                "normal": [self._serialize_traj(t) for t in split.normal],
                "anomalous": [self._serialize_traj(t) for t in split.anomalous],
                "concept_ids": split.concept_ids,
                "primitive_labels": split.primitive_labels,
            }
            torch.save(payload, str(out_dir / f"{split_name}.pt"))

        # Metadata JSON (convert numpy types for JSON serialisation)
        meta = _numpy_safe(benchmark.metadata)
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

    @staticmethod
    def _serialize_traj(traj: SemanticTrajectory) -> Dict[str, Any]:
        return {
            "episodes": traj.to_tensor_list(),
            "user_id": traj.user_id,
            "trip_id": traj.trip_id,
            "label": traj.label,
            "primitive_labels": traj.primitive_labels,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numpy_safe(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, dict):
        return {str(k): _numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="Build MobDef-Bench benchmark")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory to write benchmark splits")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing raw datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (skip real dataset download)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to build (default: all)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    builder = MobDefBenchBuilder(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    if args.synthetic:
        # Synthetic mode: build only numosim-equivalent with generated data
        datasets = ["numosim"]
        logger.info("Synthetic mode: building benchmark for dataset: numosim")
        trajectories = builder.generate_synthetic_data(n_users=200, trips_per_user=100)
        trajectories = builder._tokenize(trajectories)
        train_trajs, val_trajs, test_trajs = builder._split_by_user(trajectories)

        benchmark = Benchmark(dataset_name="numosim")
        benchmark.train.normal = train_trajs
        benchmark.val.normal = val_trajs
        benchmark.test.normal = test_trajs

        from .concept_generator import CONCEPT_DEFS, SPLIT_SEEN, SPLIT_ZS_COMP, SPLIT_ZS_FAMILY

        seen = [c for c in CONCEPT_DEFS if c.split == SPLIT_SEEN]
        zs_comp = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_COMP]
        zs_family = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_FAMILY]
        unknown = [c for c in CONCEPT_DEFS if c.split == SPLIT_UNKNOWN]

        builder._inject_anomalies(benchmark.train, train_trajs, seen)
        builder._inject_anomalies(benchmark.val, val_trajs, seen + zs_comp)
        builder._inject_anomalies(benchmark.test, test_trajs, seen + zs_comp + zs_family + unknown)

        issues = builder.validate_benchmark(benchmark)
        benchmark.metadata = {
            "dataset": "numosim",
            "seed": args.seed,
            "n_train_normal": len(benchmark.train.normal),
            "n_train_anomalous": len(benchmark.train.anomalous),
            "n_val_normal": len(benchmark.val.normal),
            "n_val_anomalous": len(benchmark.val.anomalous),
            "n_test_normal": len(benchmark.test.normal),
            "n_test_anomalous": len(benchmark.test.anomalous),
            "validation_issues": issues,
            "statistics": builder.get_statistics(benchmark),
        }
        builder._save_benchmark(benchmark)
        logger.info("Benchmark saved to %s/numosim/", args.output_dir)
        logger.info("Metadata: %s", json.dumps(_numpy_safe(benchmark.metadata), indent=2))
    else:
        datasets = args.datasets
        benchmarks = builder.build(datasets=datasets)
        logger.info("Built %d benchmark(s): %s", len(benchmarks), list(benchmarks.keys()))
        for ds_name, bm in benchmarks.items():
            logger.info("[%s] train=%d/%d  val=%d/%d  test=%d/%d  issues=%d",
                        ds_name,
                        len(bm.train.normal), len(bm.train.anomalous),
                        len(bm.val.normal), len(bm.val.anomalous),
                        len(bm.test.normal), len(bm.test.anomalous),
                        len(bm.metadata.get("validation_issues", [])))


if __name__ == "__main__":
    main()
