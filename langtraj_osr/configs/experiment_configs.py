"""Generate experiment configurations for each phase of the LangTraj-OSR evaluation.

Usage:
    from langtraj_osr.configs.experiment_configs import get_phase_configs
    configs = get_phase_configs(phase=2)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Base configuration (mirrors default.yaml)
# ---------------------------------------------------------------------------
_BASE_CONFIG: Dict[str, Any] = {
    "model": {
        "d_model": 256,
        "nhead": 4,
        "num_layers": 4,
        "dropout": 0.1,
        "max_len": 64,
        "poi_vocab_size": 64,
        "time_bins": 168,
        "dwell_bins": 16,
        "transition_types": 4,
        "n_prototypes": 8,
        "n_primitives": 10,
        "text_encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "lambda_prim": 0.5,
        "gamma_loc": 0.25,
    },
    "training": {
        "batch_size": 256,
        "lr_backbone": 1.0e-4,
        "lr_heads": 5.0e-4,
        "weight_decay": 1.0e-2,
        "epochs": 50,
        "temperature": 0.07,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "loss_weights": {
            "pair": 1.0,
            "cls": 0.5,
            "prim": 1.0,
            "para": 0.2,
            "orth": 0.05,
            "norm": 1.0,
        },
    },
    "data": {
        "max_history_trips": 50,
        "min_history_trips": 5,
    },
    "evaluation": {
        "conformal_alpha_norm": 0.05,
        "conformal_alpha_concept": 0.10,
    },
    "seeds": [42, 123, 456],
}

DATASETS = ["numosim", "geolife", "porto", "foursquare"]

BASELINES = [
    "NormOnly",
    "DSL-XL",
    "Backbone+max",
    "Backbone+cosine",
    "CLIP-Mob",
    "LangTraj-OSR",
]


def _cfg(overrides: Dict[str, Any] | None = None, **kwargs: Any) -> Dict[str, Any]:
    """Return a deep copy of base config with optional overrides applied."""
    cfg = copy.deepcopy(_BASE_CONFIG)
    if overrides:
        _deep_update(cfg, overrides)
    _deep_update(cfg, kwargs)
    return cfg


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> None:
    """Recursively update *base* dict with *updates*."""
    for key, val in updates.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], val)
        else:
            base[key] = val


# ---------------------------------------------------------------------------
# Phase generators
# ---------------------------------------------------------------------------

def _phase1_benchmark_construction() -> List[Dict[str, Any]]:
    """Phase 1: Construct MobDef-Bench datasets for all four data sources."""
    configs: List[Dict[str, Any]] = []
    for ds in DATASETS:
        configs.append(
            _cfg(
                overrides={
                    "phase": 1,
                    "phase_name": "benchmark_construction",
                    "dataset": ds,
                    "task": "build_benchmark",
                    "description": f"Build MobDef-Bench for {ds}",
                }
            )
        )
    return configs


def _phase2_sanity_baselines() -> List[Dict[str, Any]]:
    """Phase 2: Sanity baselines on NUMOSIM (NormOnly, DSL-XL, Backbone+max)."""
    configs: List[Dict[str, Any]] = []
    sanity_baselines = ["NormOnly", "DSL-XL", "Backbone+max"]
    for baseline in sanity_baselines:
        for seed in _BASE_CONFIG["seeds"]:
            cfg = _cfg(
                overrides={
                    "phase": 2,
                    "phase_name": "sanity_baselines",
                    "dataset": "numosim",
                    "baseline": baseline,
                    "seed": seed,
                    "description": f"Sanity check: {baseline} on NUMOSIM (seed={seed})",
                }
            )
            # Baseline-specific modifications
            if baseline == "NormOnly":
                cfg["model"]["disable_language"] = True
                cfg["model"]["disable_primitives"] = True
                cfg["training"]["loss_weights"]["pair"] = 0.0
                cfg["training"]["loss_weights"]["cls"] = 0.0
                cfg["training"]["loss_weights"]["prim"] = 0.0
            elif baseline == "DSL-XL":
                cfg["model"]["use_dsl"] = True
                cfg["model"]["disable_language"] = True
            elif baseline == "Backbone+max":
                cfg["model"]["disable_conformal"] = True
                cfg["model"]["aggregation"] = "max"
            configs.append(cfg)
    return configs


def _phase3_full_model() -> List[Dict[str, Any]]:
    """Phase 3: Full LangTraj-OSR model on NUMOSIM and GeoLife."""
    configs: List[Dict[str, Any]] = []
    for ds in ["numosim", "geolife"]:
        for seed in _BASE_CONFIG["seeds"]:
            configs.append(
                _cfg(
                    overrides={
                        "phase": 3,
                        "phase_name": "full_model",
                        "dataset": ds,
                        "baseline": "LangTraj-OSR",
                        "seed": seed,
                        "description": f"Full LangTraj-OSR on {ds} (seed={seed})",
                    }
                )
            )
    return configs


def _phase4_zero_shot_composition() -> List[Dict[str, Any]]:
    """Phase 4: Zero-shot composition split evaluation."""
    configs: List[Dict[str, Any]] = []
    for ds in ["numosim", "geolife"]:
        for seed in _BASE_CONFIG["seeds"]:
            configs.append(
                _cfg(
                    overrides={
                        "phase": 4,
                        "phase_name": "zero_shot_composition",
                        "dataset": ds,
                        "baseline": "LangTraj-OSR",
                        "seed": seed,
                        "split": "composition",
                        "description": (
                            f"Zero-shot composition split on {ds} (seed={seed})"
                        ),
                    }
                )
            )
    return configs


def _phase5_zero_shot_family() -> List[Dict[str, Any]]:
    """Phase 5: Zero-shot family split evaluation."""
    configs: List[Dict[str, Any]] = []
    for ds in ["numosim", "geolife"]:
        for seed in _BASE_CONFIG["seeds"]:
            configs.append(
                _cfg(
                    overrides={
                        "phase": 5,
                        "phase_name": "zero_shot_family",
                        "dataset": ds,
                        "baseline": "LangTraj-OSR",
                        "seed": seed,
                        "split": "family",
                        "description": (
                            f"Zero-shot family split on {ds} (seed={seed})"
                        ),
                    }
                )
            )
    return configs


def _phase6_full_baseline_suite() -> List[Dict[str, Any]]:
    """Phase 6: All datasets x all baselines."""
    configs: List[Dict[str, Any]] = []
    for ds in DATASETS:
        for baseline in BASELINES:
            for seed in _BASE_CONFIG["seeds"]:
                cfg = _cfg(
                    overrides={
                        "phase": 6,
                        "phase_name": "full_baseline_suite",
                        "dataset": ds,
                        "baseline": baseline,
                        "seed": seed,
                        "description": (
                            f"{baseline} on {ds} (seed={seed})"
                        ),
                    }
                )
                # Apply baseline-specific config tweaks
                if baseline == "NormOnly":
                    cfg["model"]["disable_language"] = True
                    cfg["model"]["disable_primitives"] = True
                    cfg["training"]["loss_weights"]["pair"] = 0.0
                    cfg["training"]["loss_weights"]["cls"] = 0.0
                    cfg["training"]["loss_weights"]["prim"] = 0.0
                elif baseline == "DSL-XL":
                    cfg["model"]["use_dsl"] = True
                    cfg["model"]["disable_language"] = True
                elif baseline == "Backbone+max":
                    cfg["model"]["disable_conformal"] = True
                    cfg["model"]["aggregation"] = "max"
                elif baseline == "Backbone+cosine":
                    cfg["model"]["disable_conformal"] = True
                    cfg["model"]["aggregation"] = "cosine"
                elif baseline == "CLIP-Mob":
                    cfg["model"]["use_clip_style"] = True
                    cfg["model"]["disable_primitives"] = True
                configs.append(cfg)
    return configs


def _phase7_cross_city_transfer() -> List[Dict[str, Any]]:
    """Phase 7: Cross-city transfer (NYC <-> Tokyo)."""
    configs: List[Dict[str, Any]] = []
    transfer_pairs = [
        ("foursquare_nyc", "foursquare_tokyo"),
        ("foursquare_tokyo", "foursquare_nyc"),
    ]
    for train_city, eval_city in transfer_pairs:
        for seed in _BASE_CONFIG["seeds"]:
            configs.append(
                _cfg(
                    overrides={
                        "phase": 7,
                        "phase_name": "cross_city_transfer",
                        "train_dataset": train_city,
                        "eval_dataset": eval_city,
                        "baseline": "LangTraj-OSR",
                        "seed": seed,
                        "description": (
                            f"Transfer {train_city} -> {eval_city} (seed={seed})"
                        ),
                    }
                )
            )
    return configs


def _phase8_robustness() -> List[Dict[str, Any]]:
    """Phase 8: Robustness to paraphrase and analyst noise."""
    configs: List[Dict[str, Any]] = []
    robustness_tests = [
        "paraphrase_5x",
        "paraphrase_10x",
        "analyst_noise_low",
        "analyst_noise_high",
        "typo_injection",
    ]
    for ds in ["numosim", "geolife"]:
        for test in robustness_tests:
            for seed in _BASE_CONFIG["seeds"]:
                configs.append(
                    _cfg(
                        overrides={
                            "phase": 8,
                            "phase_name": "robustness",
                            "dataset": ds,
                            "baseline": "LangTraj-OSR",
                            "robustness_test": test,
                            "seed": seed,
                            "description": (
                                f"Robustness ({test}) on {ds} (seed={seed})"
                            ),
                        }
                    )
                )
    return configs


def _phase9_ablations() -> List[Dict[str, Any]]:
    """Phase 9: All ablation studies A1-A11."""
    ablation_specs: List[Dict[str, Any]] = [
        {
            "id": "A1",
            "name": "no_language",
            "description": "No language (normality only)",
            "overrides": {
                "model": {
                    "disable_language": True,
                    "disable_primitives": True,
                },
                "training": {
                    "loss_weights": {
                        "pair": 0.0,
                        "cls": 0.0,
                        "prim": 0.0,
                        "para": 0.0,
                    },
                },
            },
        },
        {
            "id": "A2",
            "name": "no_primitive_head",
            "description": "Language but no primitive head",
            "overrides": {
                "model": {"disable_primitives": True},
                "training": {"loss_weights": {"prim": 0.0}},
            },
        },
        {
            "id": "A3",
            "name": "dsl_instead_of_language",
            "description": "DSL instead of free-text language",
            "overrides": {
                "model": {"use_dsl": True, "disable_language": True},
            },
        },
        {
            "id": "A4",
            "name": "no_user_history",
            "description": "No user-history module",
            "overrides": {
                "model": {"disable_user_history": True},
                "training": {"loss_weights": {"norm": 0.0}},
            },
        },
        {
            "id": "A5",
            "name": "cohort_history",
            "description": "Cohort history instead of per-user",
            "overrides": {
                "model": {"use_cohort_history": True},
            },
        },
        {
            "id": "A6",
            "name": "no_paraphrase_loss",
            "description": "No paraphrase consistency loss",
            "overrides": {
                "training": {"loss_weights": {"para": 0.0}},
            },
        },
        {
            "id": "A7",
            "name": "no_conformal_reject",
            "description": "No conformal reject (revert to max-softmax)",
            "overrides": {
                "model": {"disable_conformal": True},
            },
        },
        {
            "id": "A8",
            "name": "opaque_representation",
            "description": "Opaque (unfactored) representation",
            "overrides": {
                "model": {"use_opaque_representation": True},
            },
        },
        {
            "id": "A9",
            "name": "no_orthogonality",
            "description": "No L_orth loss",
            "overrides": {
                "training": {"loss_weights": {"orth": 0.0}},
            },
        },
        {
            "id": "A10",
            "name": "random_text_embeddings",
            "description": "Random text embeddings instead of SentenceTransformer",
            "overrides": {
                "model": {"use_random_text_embeddings": True},
            },
        },
        {
            "id": "A11",
            "name": "bow_text_features",
            "description": "BoW/TF-IDF text features instead of SentenceTransformer",
            "overrides": {
                "model": {"use_bow_text": True},
            },
        },
    ]

    configs: List[Dict[str, Any]] = []
    for ablation in ablation_specs:
        for ds in ["numosim", "geolife"]:
            for seed in _BASE_CONFIG["seeds"]:
                cfg = _cfg(overrides=ablation["overrides"])
                cfg["phase"] = 9
                cfg["phase_name"] = "ablations"
                cfg["dataset"] = ds
                cfg["baseline"] = "LangTraj-OSR"
                cfg["ablation_id"] = ablation["id"]
                cfg["ablation_name"] = ablation["name"]
                cfg["seed"] = seed
                cfg["description"] = (
                    f"Ablation {ablation['id']}: {ablation['description']} "
                    f"on {ds} (seed={seed})"
                )
                configs.append(cfg)
    return configs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PHASE_GENERATORS = {
    1: _phase1_benchmark_construction,
    2: _phase2_sanity_baselines,
    3: _phase3_full_model,
    4: _phase4_zero_shot_composition,
    5: _phase5_zero_shot_family,
    6: _phase6_full_baseline_suite,
    7: _phase7_cross_city_transfer,
    8: _phase8_robustness,
    9: _phase9_ablations,
}

PHASE_NAMES = {
    1: "Benchmark construction",
    2: "Sanity baselines (NUMOSIM)",
    3: "Full model (NUMOSIM + GeoLife)",
    4: "Zero-shot composition split",
    5: "Zero-shot family split",
    6: "Full baseline suite",
    7: "Cross-city transfer (NYC <-> Tokyo)",
    8: "Robustness (paraphrase + analyst noise)",
    9: "All ablations (A1-A11)",
}

# Go/no-go criteria per phase
GO_NOGO_CRITERIA: Dict[int, Dict[str, Any]] = {
    2: {
        "metric": "auroc",
        "threshold": 0.55,
        "condition": "all baselines exceed random-chance AUROC",
    },
    3: {
        "metric": "auroc",
        "threshold": 0.70,
        "condition": "LangTraj-OSR AUROC >= 0.70 on NUMOSIM",
    },
    4: {
        "metric": "auroc",
        "threshold": 0.60,
        "condition": "Zero-shot composition AUROC >= 0.60",
    },
    5: {
        "metric": "auroc",
        "threshold": 0.55,
        "condition": "Zero-shot family AUROC >= 0.55",
    },
    6: {
        "metric": "auroc_improvement",
        "threshold": 0.03,
        "condition": "LangTraj-OSR beats best baseline by >= 3% AUROC",
    },
    7: {
        "metric": "transfer_drop",
        "threshold": 0.15,
        "condition": "Cross-city transfer drop <= 15%",
    },
    8: {
        "metric": "worst_paraphrase_gap",
        "threshold": 0.05,
        "condition": "Worst paraphrase gap <= 5%",
    },
    9: {
        "metric": "ablation_drop_max",
        "threshold": 0.20,
        "condition": "No single ablation drops AUROC by more than 20%",
    },
}


def get_phase_configs(phase: int) -> List[Dict[str, Any]]:
    """Generate experiment configurations for a given phase.

    Parameters
    ----------
    phase : int
        Experiment phase number (1-9).

    Returns
    -------
    list[dict]
        List of configuration dicts, one per experiment run.

    Raises
    ------
    ValueError
        If *phase* is not in the range 1-9.
    """
    if phase not in _PHASE_GENERATORS:
        raise ValueError(
            f"Unknown phase {phase}. Valid phases: {sorted(_PHASE_GENERATORS.keys())}"
        )
    return _PHASE_GENERATORS[phase]()


def get_go_nogo(phase: int) -> Dict[str, Any] | None:
    """Return go/no-go criteria for a phase, or None if none defined."""
    return GO_NOGO_CRITERIA.get(phase)
