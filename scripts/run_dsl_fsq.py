"""
Standalone DSL-XL evaluation on Foursquare (Tokyo / NYC).
Mirrors the foursquare loading in train.py and the baseline runner in run_baseline.py.

Usage:
    python run_dsl_fsq.py --city tokyo --seed 42
    python run_dsl_fsq.py --city nyc   --seed 42
"""
import argparse
import hashlib
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = "/home/hello/ouyangqi"
sys.path.insert(0, BASE_DIR)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Foursquare data loading (mirrors train.py exactly)
# ---------------------------------------------------------------------------
CHECKIN_RE = re.compile(
    r"At (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), user (\d+) visited POI id (\d+)"
    r" which is a .+? and has Category id (\d+)\."
)


def _discretize_dwell(dwell_min: float) -> int:
    breaks = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 360, 420, 480]
    for i, b in enumerate(breaks):
        if dwell_min <= b:
            return i
    return len(breaks)


def parse_foursquare_parquet(path: str):
    from langtraj_osr.core.episode import SemanticEpisode, SemanticTrajectory

    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)

    user_checkins: Dict[str, list] = defaultdict(list)
    for txt in df["inputs"]:
        for m in CHECKIN_RE.finditer(txt):
            ts_str = m.group(1)
            uid = m.group(2)
            poi_id = int(m.group(3))
            cat_id = int(m.group(4))
            ts = pd.Timestamp(ts_str)
            user_checkins[uid].append((ts, poi_id, cat_id))

    trajs = []
    for uid, checkins in user_checkins.items():
        checkins.sort(key=lambda x: x[0])
        by_date: Dict[str, list] = defaultdict(list)
        for ts, poi, cat in checkins:
            by_date[ts.date().isoformat()].append((ts, poi, cat))

        avg_day_len = float(sum(len(v) for v in by_date.values()) / max(len(by_date), 1))

        for date_str, day_checkins in by_date.items():
            if len(day_checkins) < 3:
                continue
            eps = []
            for i, (ts, poi, cat) in enumerate(day_checkins):
                if i + 1 < len(day_checkins):
                    dwell_min = (day_checkins[i + 1][0] - ts).total_seconds() / 60.0
                else:
                    dwell_min = 30.0
                dwell_min = min(max(dwell_min, 1.0), 480.0)
                trans = 0 if dwell_min < 20 else (2 if dwell_min < 60 else 1)
                zone = int(hashlib.md5(str(poi).encode()).hexdigest()[:8], 16) % (2**31)
                time_bin = ts.hour * 7 + ts.dayofweek
                dwell_bin = _discretize_dwell(dwell_min)
                tlc = float(len(day_checkins)) / max(avg_day_len, 1.0)
                tlc = min(tlc, 20.0)
                eps.append(
                    SemanticEpisode(
                        zone_id=zone,
                        poi_role=int(cat) % 64,
                        time_bin=time_bin,
                        dwell_bin=dwell_bin,
                        transition_type=trans,
                        trip_length_change=round(tlc, 4),
                        event_flag=0,
                        companion_flag=0,
                    )
                )
            trajs.append(
                SemanticTrajectory(
                    episodes=eps,
                    user_id=uid,
                    trip_id=f"{uid}_{date_str}",
                    label=0,
                )
            )
    logger.info("Parsed %d trips from %d users at %s", len(trajs), len(user_checkins), path)
    return trajs


def build_foursquare_benchmark(city: str, seed: int):
    import tempfile
    from langtraj_osr.benchmark.benchmark_builder import (
        MobDefBenchBuilder,
        Benchmark,
        CONCEPT_DEFS,
        SPLIT_SEEN,
        SPLIT_ZS_COMP,
        SPLIT_ZS_FAMILY,
        SPLIT_UNKNOWN,
    )

    train_path = f"{BASE_DIR}/data/foursquare_{city}/train.parquet"
    # also load test parquet for extra data
    test_path = f"{BASE_DIR}/data/foursquare_{city}/test.parquet"

    raw_trajs = parse_foursquare_parquet(train_path)
    try:
        raw_extra = parse_foursquare_parquet(test_path)
        raw_trajs = raw_trajs + raw_extra
        logger.info("Combined train+test parquet: %d trips", len(raw_trajs))
    except Exception:
        pass

    with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as to:
        builder = MobDefBenchBuilder(data_dir=td, output_dir=to, seed=seed)
        raw_trajs = builder._tokenize(raw_trajs)
        tr, va, te = builder._split_by_user(raw_trajs)
        bench = Benchmark(dataset_name=f"foursquare_{city}")
        bench.train.normal = tr
        bench.val.normal = va
        bench.test.normal = te

        sc = [c for c in CONCEPT_DEFS if c.split == SPLIT_SEEN]
        zc = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_COMP]
        zf = [c for c in CONCEPT_DEFS if c.split == SPLIT_ZS_FAMILY]
        uc = [c for c in CONCEPT_DEFS if c.split == SPLIT_UNKNOWN]

        builder._inject_anomalies(bench.train, tr, sc + zc + zf)
        builder._inject_anomalies(bench.val, va, sc + zc)
        builder._inject_anomalies(bench.test, te, sc + zc + zf + uc)

    def _comb(s):
        return s.normal + s.anomalous

    train_trajs = _comb(bench.train)
    val_trajs = _comb(bench.val)
    test_trajs = _comb(bench.test)

    logger.info(
        "Benchmark: train=%d (norm=%d, anom=%d), val=%d, test=%d",
        len(train_trajs),
        len(bench.train.normal),
        len(bench.train.anomalous),
        len(val_trajs),
        len(test_trajs),
    )
    return train_trajs, val_trajs, test_trajs


# ---------------------------------------------------------------------------
# Dataset (mirrors SyntheticBaselineDataset in run_baseline.py)
# ---------------------------------------------------------------------------
MAX_LEN = 64


def traj_to_episodes(traj, max_len: int = MAX_LEN) -> Dict[str, torch.Tensor]:
    eps_lists = [ep.to_list() for ep in traj.episodes[:max_len]]
    while len(eps_lists) < max_len:
        eps_lists.append([0.0] * 8)
    ep_tensor = torch.tensor(eps_lists, dtype=torch.float32)
    return {
        "zone_id": ep_tensor[:, 0].long(),
        "poi_role": ep_tensor[:, 1].long().clamp(0, 63),
        "time_bin": ep_tensor[:, 2].long().clamp(0, 167),
        "dwell_bin": ep_tensor[:, 3].long().clamp(0, 15),
        "transition_type": ep_tensor[:, 4].long().clamp(0, 3),
        "trip_length_change": ep_tensor[:, 5].float(),
        "event_flag": ep_tensor[:, 6].long().clamp(0, 1),
        "companion_flag": ep_tensor[:, 7].long().clamp(0, 1),
    }


DUMMY_PROTOS = {
    "mu": torch.zeros(8, 256),
    "sigma": torch.ones(8, 256),
    "pi": torch.ones(8) / 8,
}


class TrajDataset(Dataset):
    def __init__(self, trajs: list) -> None:
        self.trajs = trajs

    def __len__(self) -> int:
        return len(self.trajs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        t = self.trajs[idx]
        return {
            "episodes": traj_to_episodes(t),
            "label": torch.tensor(t.label, dtype=torch.long),
            "user_prototypes": DUMMY_PROTOS,
        }


def collate_fn(batch: list) -> Dict[str, Any]:
    episodes: Dict[str, list] = {}
    labels: list = []
    for s in batch:
        for k, v in s["episodes"].items():
            episodes.setdefault(k, []).append(v)
        labels.append(s["label"])
    return {
        "episodes": {k: torch.stack(v) for k, v in episodes.items()},
        "labels": torch.stack(labels),
        "user_prototypes": {
            "mu": torch.zeros(len(batch), 8, 256),
            "sigma": torch.ones(len(batch), 8, 256),
            "pi": torch.ones(len(batch), 8) / 8,
        },
    }


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_dslxl(
    city: str,
    seed: int,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> Dict[str, float]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_trajs, val_trajs, test_trajs = build_foursquare_benchmark(city, seed)

    from langtraj_osr.baselines.dsl_xl import DSLXLModel
    from langtraj_osr.core.concepts import get_concept_ids_for_split

    model = DSLXLModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # DSL slots — use zeros (default, as in run_baseline.py fallback)
    dsl_slots = torch.zeros(25, 12, dtype=torch.long, device=device)

    train_loader = DataLoader(
        TrajDataset(train_trajs), batch_size=batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn, pin_memory=True,
    )

    # Training loop
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            episodes = {k: v.to(device) for k, v in batch["episodes"].items()}
            labels = batch["labels"].to(device)
            user_protos = {k: v.to(device) for k, v in batch["user_prototypes"].items()}
            loss = model.compute_loss(episodes, dsl_slots, user_protos, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if ep % 10 == 0:
            logger.info("Epoch %d/%d  loss=%.4f", ep, epochs, total_loss / len(train_loader))

    # Evaluation
    model.eval()
    test_loader = DataLoader(
        TrajDataset(test_trajs), batch_size=256, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
    )

    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            episodes = {k: v.to(device) for k, v in batch["episodes"].items()}
            labels = batch["labels"].cpu().numpy()
            user_protos = {k: v.to(device) for k, v in batch["user_prototypes"].items()}
            def_scores, E_norm = model.predict(episodes, dsl_slots, user_protos)
            # anomaly score = max concept score
            scores = def_scores.max(dim=-1).values.cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels)

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    from sklearn.metrics import roc_auc_score

    def split_auroc(split_name: str) -> float:
        ids = list(get_concept_ids_for_split(split_name))
        mask = (labels == 0) | np.isin(labels, ids)
        if mask.sum() == 0:
            return float("nan")
        y = (labels[mask] != 0).astype(int)
        if y.sum() == 0 or (1 - y).sum() == 0:
            return float("nan")
        return float(roc_auc_score(y, scores[mask]))

    try:
        overall = float(roc_auc_score((labels != 0).astype(int), scores))
    except Exception:
        overall = float("nan")

    results = {
        "city": city,
        "seed": seed,
        "overall_auroc": overall,
        "A_seen_auroc": split_auroc("seen"),
        "A_zs_comp_auroc": split_auroc("zs_comp"),
        "A_zs_family_auroc": split_auroc("zs_family"),
        "A_unknown_auroc": split_auroc("unknown"),
    }

    logger.info("=== DSL-XL Foursquare %s seed %d ===", city.upper(), seed)
    for k, v in results.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)

    out_dir = Path(BASE_DIR) / f"results/dsl_fsq_{city}/seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: %s", out_path)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, choices=["tokyo", "nyc"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train_dslxl(args.city, args.seed, args.epochs, args.batch_size, args.lr)
