# LangTraj-OSR

**Language-Guided Open-Set Anomaly Detection and Concept Assignment in Semantic Trajectories**

Code and experiment results for the KDD 2026 submission. LangTraj-OSR enables analysts to define anomaly concepts in natural language and assigns each flagged trajectory to the matching definition, with calibrated rejection of unknown anomaly types.

---

## Repository Structure

```
code/
├── langtraj_osr/                        # Main Python package
│   ├── train.py                         # 3-stage training pipeline
│   ├── evaluate.py                      # Full evaluation with all metrics
│   ├── core/
│   │   ├── concepts.py                  # 25 anomaly concept definitions + split assignments
│   │   ├── episode.py                   # SemanticEpisode & SemanticTrajectory dataclasses
│   │   ├── dataset.py                   # MobDefBenchDataModule + data loaders
│   │   ├── tokenizer.py                 # GPS/check-in -> SemanticEpisode conversion
│   │   └── utils.py                     # Metrics, logging, reproducibility
│   ├── models/
│   │   ├── langtraj_osr.py              # Full LangTraj-OSR model
│   │   ├── episode_encoder.py           # (B,L,8) -> (B,L,256) episode embeddings
│   │   ├── trajectory_encoder.py        # 4-layer Transformer -> (B,256)
│   │   ├── definition_encoder.py        # Frozen SentenceTransformer text encoder
│   │   ├── user_history.py              # Per-user GMM prototype banks + normality energy
│   │   ├── primitive_head.py            # 10-dim binary primitive predictions
│   │   ├── losses.py                    # L_pair, L_cls, L_prim, L_para, L_orth, L_repel
│   │   └── conformal.py                 # Split-conformal calibration (Stage A + B)
│   ├── baselines/
│   │   ├── run_baseline.py              # Unified baseline runner
│   │   ├── norm_only.py                 # B1: Pure normality detector
│   │   ├── dsl_xl.py                    # B2: 12-slot domain-specific language
│   │   ├── nl2dsl.py                    # B3: NL->DSL compilation
│   │   ├── canonical_json.py            # B4: Structured JSON definitions
│   │   ├── direct_text.py               # B5: Direct text matching
│   │   ├── lm_tad.py                    # B6: Autoregressive next-episode prediction
│   │   ├── atrom_ossl.py                # B7: Open-set learner (EVT)
│   │   └── backbone_max.py              # B8: Backbone + max-score aggregation
│   ├── benchmark/
│   │   ├── benchmark_builder.py         # MobDef-Bench 7-step construction pipeline
│   │   ├── concept_generator.py         # Compose 10 primitives into 25 concepts
│   │   ├── interventions.py             # 10 executable anomaly injection operators
│   │   └── synthetic_data.py            # Synthetic trajectory generator
│   ├── evaluation/
│   │   ├── metrics.py                   # AUROC, AUPRC, H-score, ECE, calibration
│   │   └── visualize.py                 # Result visualization & plotting
│   ├── configs/
│   │   ├── experiment_configs.py        # Phase-specific experiment configs
│   │   └── default.yaml                 # Default hyperparameters
│   └── scripts/
│       ├── run_phase.py                 # Execute single experiment phase
│       ├── ablation_runner.py           # Run ablation variants
│       ├── summarize_results.py         # Aggregate results across phases
│       └── download_assets.py           # Download datasets & models
├── scripts/                             # Standalone experiment scripts
│   ├── run_ablation_typed_v2_bs64.py    # 8-variant ablation (final version)
│   ├── run_baselines_porto.py           # Porto baseline suite
│   ├── run_primitive_oracle.py          # Primitive-only oracle (circularity check)
│   ├── eval_multiseed_foursquare.py     # Tokyo/NYC multi-seed evaluation
│   ├── run_porto_case_study.py          # Qualitative case study
│   ├── run_diagnostic_plots.py          # Figure generation
│   ├── run_retrieval_baselines.py       # Retrieval-based baselines
│   ├── run_typed_foursquare_and_centroid.py  # Foursquare typed evaluation
│   ├── robustness_exp.py               # Definition robustness experiments
│   ├── bank_scaling_exp.py             # Concept bank scaling
│   ├── run_synth_prim_fix.py           # Synthetic primitive fix
│   ├── run_porto_prim_fix.py           # Porto primitive fix
│   ├── preprocess_porto.py             # Porto GPS -> semantic episodes
│   ├── convert_porto_to_pickle.py      # Porto parquet -> pickle
│   ├── run_porto_experiment.py         # Porto full experiment
│   ├── run_dsl_fsq.py                  # DSL-XL on Foursquare
│   ├── ablation_typed.py               # Typed ablation
│   ├── nyc_stratify.py                 # NYC stratified analysis
│   └── download_for_offline.py         # Download models for offline use
├── data/
│   ├── porto/                           # Porto taxi GPS data
│   │   └── porto_taxi.py               # Data loader
│   ├── geolife/                         # GeoLife trajectory data
│   │   └── geolife.py                  # Data loader
│   ├── foursquare_tokyo/               # Tokyo check-in data
│   └── foursquare_nyc/                 # NYC check-in data
├── results/                             # All experiment results (see below)
├── run_all_experiments.sh               # One-click experiment orchestrator
├── setup.py                             # Package installation
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## Requirements

- Python >= 3.9
- PyTorch >= 2.0 (with CUDA support)
- GPU: NVIDIA RTX 3090 / 4090 or equivalent (16+ GB VRAM)
- sentence-transformers (all-MiniLM-L6-v2)

## Installation

```bash
cd code
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

### 1. Synthetic Data (No Downloads Required)

```bash
# Train on synthetic MobDef-Bench data (single seed)
python -m langtraj_osr.train --use_synthetic --seed 42 --epochs 50 --batch_size 128 \
    --output_dir results/synthetic/seed_42

# Evaluate
python -m langtraj_osr.evaluate \
    --checkpoint results/synthetic/seed_42/best_model.pt \
    --dataset synthetic --seed 42
```

### 2. Porto Taxi GPS Data

```bash
# Preprocess Porto GPS traces into semantic episodes
python scripts/preprocess_porto.py --input data/porto/train.csv.zip \
    --output data/porto/porto_processed.pkl

# Train
python -m langtraj_osr.train --use_porto_real \
    --porto_parquet data/porto/porto_processed.pkl \
    --seed 42 --epochs 50 --batch_size 128 \
    --output_dir results/porto/seed_42

# Evaluate
python -m langtraj_osr.evaluate \
    --checkpoint results/porto/seed_42/best_model.pt \
    --dataset porto --seed 42
```

### 3. Foursquare Check-in Data (Tokyo / NYC)

```bash
# Train on Tokyo
python -m langtraj_osr.train --use_foursquare \
    --foursquare_dir data/foursquare_tokyo \
    --seed 42 --epochs 50 --batch_size 128 \
    --output_dir results/tokyo/seed_42

# Train on NYC
python -m langtraj_osr.train --use_foursquare \
    --foursquare_dir data/foursquare_nyc \
    --seed 42 --epochs 50 --batch_size 128 \
    --output_dir results/nyc/seed_42
```

### 4. Full Experiment Suite (All Datasets, 3 Seeds)

```bash
# Run everything: benchmark, training, baselines, ablations
bash run_all_experiments.sh --gpu 0 --seeds "42 123 456"

# Synthetic only (fast, no data downloads)
bash run_all_experiments.sh --gpu 0 --synthetic

# Resume from a specific phase
bash run_all_experiments.sh --gpu 0 --from_phase 6
```

---

## Training Pipeline

LangTraj-OSR uses a 3-stage training pipeline:

| Stage | Description | Epochs | Key Components |
|-------|-------------|--------|----------------|
| 1 | Self-supervised pretraining (masked episode prediction) | 10 | Cross-entropy on masked poi_role, time_bin, dwell_bin, transition_type |
| 2 | User routine bank fitting (GMM over normal embeddings) | - | K-means (K=8 prototypes per user) + GMM |
| 3 | Concept alignment (contrastive + repulsion) | 50 | L_pair + L_repel + L_cls + L_prim + L_para + L_orth |

After Stage 3, user routine banks are **re-fitted** with the final encoder for calibrated normality scoring.

### Loss Components (Stage 3)

| Loss | Weight | Purpose |
|------|--------|---------|
| L_pair | 1.0 | InfoNCE contrastive (trajectory vs definition) |
| L_repel | 0.3 | Push normal embeddings away from concept centers |
| L_cls | 0.5 | Cross-entropy for seen-concept classification |
| L_prim | 1.0 | Multi-label BCE on 10 primitive indicators |
| L_para | 0.2 | Paraphrase consistency (KL divergence) |
| L_orth | 0.05 | Orthogonality between deviation features and concept scores |

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dim | 256 |
| Transformer layers | 4 |
| Attention heads | 4 |
| Max sequence length | 64 |
| User prototypes (K) | 8 |
| Temperature (tau) | 0.07 |
| Conformal alpha | 0.05 |
| Text encoder | all-MiniLM-L6-v2 (frozen) |
| Batch size | 128 (64 for ablation) |

---

## Baselines

8 baselines are implemented in `langtraj_osr/baselines/`:

| ID | Method | Description |
|----|--------|-------------|
| B1 | NormOnly | Pure normality-based energy detector (no concept info) |
| B2 | DSL-XL | 12-slot domain-specific language definitions |
| B3 | NL2DSL | NL -> DSL rule-based parser |
| B4 | CanonicalJSON | Structured JSON concept definitions |
| B5 | DirectText | Direct text embedding matching |
| B6 | LM-TAD | Autoregressive next-episode prediction |
| B7 | ATROM-OSSL | Open-set learner with Extreme Value Theory |
| B8 | BackboneMax | Backbone + max(language, novelty) aggregation |

### Running Baselines

```bash
# Run all baselines on synthetic data
python -m langtraj_osr.baselines.run_baseline --method all --dataset synthetic --seed 42

# Run a specific baseline
python -m langtraj_osr.baselines.run_baseline --method dsl_xl --dataset synthetic --seed 42

# Run Porto baseline suite (multi-seed)
python scripts/run_baselines_porto.py

# Run DSL-XL on Foursquare
python scripts/run_dsl_fsq.py
```

---

## Ablation Studies

8 ablation variants to isolate each component's contribution:

| Variant | Modification | Key Finding |
|---------|-------------|-------------|
| full | No change (reference) | AUROC 0.995, Top-1 0.955 |
| -L_repel | Remove repulsion loss | AUROC -8.3pp (most impactful) |
| -L_cls | Remove classification loss | Macro-F1 -17.3pp (fairness loss) |
| -L_prim | Remove primitive loss | Marginal effect |
| -L_para | Remove paraphrase loss | Marginal effect |
| -L_orth | Remove orthogonality loss | Marginal effect |
| random_bank | Random concept embeddings | Seen Top-1 preserved, ZS destroyed |
| shuffle | Shuffled definition assignments | Top-1 0.086 (detection preserved, typing destroyed) |

```bash
# Run all 8 ablation variants
python scripts/run_ablation_typed_v2_bs64.py
```

---

## Special Experiments

### Primitive-Only Oracle (Circularity Check)

Tests whether primitive indicators alone can perform concept assignment (without language). Result: **Top-1 = 0.003** vs language's 0.955, refuting benchmark circularity.

```bash
python scripts/run_primitive_oracle.py
```

### Robustness & Paraphrase Sensitivity

```bash
python scripts/robustness_exp.py
```

### Cross-City Transfer

```bash
# Train on Tokyo, evaluate on NYC (and vice versa)
# Results in results/cross_city/
```

### Porto Case Study

Qualitative analysis of detected anomalies on real Porto taxi trajectories.

```bash
python scripts/run_porto_case_study.py
```

### Multi-Seed Foursquare Evaluation

```bash
python scripts/eval_multiseed_foursquare.py
```

---

## MobDef-Bench

25 anomaly concepts composed from 10 atomic primitives across 4 evaluation splits:

| Split | Concept IDs | Count | Description |
|-------|-------------|-------|-------------|
| A_seen | 1-12 | 12 | Definitions + labeled training examples |
| A_zs-comp | 13-18 | 6 | Definitions only; novel primitive compositions |
| A_zs-fam | 19-22 | 4 | Definitions only; held-out operator family |
| A_unknown | 23-25 | 3 | No definitions, no labels; must be rejected |

### 10 Primitive Operators

time_shift, role_swap, destination_sub, detour, missing_stop, dwell_inflate, order_permute, event_conflict, companion_anomaly, spatial_deviation

### Datasets

| Dataset | Type | Source |
|---------|------|--------|
| Synthetic (NUMOSIM) | Generated | MobDef-Bench builder |
| Porto | Real GPS | Porto taxi trajectories |
| Foursquare Tokyo | Real check-in | Foursquare venue check-ins |
| Foursquare NYC | Real check-in | Foursquare venue check-ins |

---

## Evaluation Metrics

| Category | Metrics |
|----------|---------|
| Binary detection | AUROC, AUPRC, FPR@95TPR |
| Typed open-set | Top-1 Accuracy, Macro-F1, OSCR |
| Calibration | ECE, Brier score, NLL, conformal coverage |
| Per-split | All above computed per split (seen, zs-comp, zs-fam, unknown) |

---

## Main Results (3-seed mean)

### Binary Detection (AUROC)

| Dataset | Overall | A_seen | A_zs-comp | A_zs-fam | A_unknown |
|---------|---------|--------|-----------|----------|-----------|
| Synthetic | 0.995 | 0.998 | 0.988 | 0.999 | 0.993 |
| Porto | 0.995 | 0.998 | 0.988 | 0.999 | 0.993 |
| Tokyo | 0.977 +/- 0.031 | - | - | - | - |
| NYC | 0.883 +/- 0.174 | - | - | - | - |

### Typed Concept Assignment (Seen Concepts)

| Dataset | Top-1 Acc | Macro-F1 |
|---------|-----------|----------|
| Synthetic | 0.947 +/- 0.006 | 0.947 +/- 0.006 |
| Porto | 0.655 +/- 0.008 | 0.655 +/- 0.008 |
| Tokyo | 0.931 +/- 0.089 | - |
| NYC | 0.781 +/- 0.267 | - |

---

## Results Directory Guide

All experiment outputs are in `results/`:

```
results/
├── main/                        # Final main results (v6)
│   ├── synthetic_seed{42,123,456}.json
│   ├── porto_seed{42,123,456}.json
│   ├── tokyo_seed{42,123,456}.json
│   └── nyc_seed{42,123,456}.json
├── baselines_synthetic/         # 8 synthetic baselines + retrieval
│   ├── {norm_only,dsl_xl,...}_s42.json
│   ├── dsl_xl_s{123,456}.json
│   └── retrieval_baselines.json
├── baselines_porto/             # Porto baselines (multi-seed)
│   ├── summary_seed{42,123,456}.json
│   ├── {norm_only,dsl_xl,lm_tad}_s{42,123,456}.json
│   └── {nl2dsl,canonical_json,atrom_ossl}_s42.json
├── baselines_foursquare/        # DSL-XL on Tokyo/NYC
│   ├── dsl_xl_tokyo_s42.json
│   └── dsl_xl_nyc_s42.json
├── ablation/                    # 8-variant ablation study
│   ├── ablation_typed_v2_results.json   # Aggregated
│   ├── {no_repel,no_cls,...}_s42.json   # Per-variant
│   ├── shuffled_s42.json                # Language-disabled control
│   └── random_bank_s42.json             # Random concept bank
├── typed_metrics/               # Typed evaluation details
│   ├── multiseed_typed_results.json
│   ├── multiseed_foursquare_results.json
│   ├── synth_prim_fix_typed.json
│   ├── porto_prim_fix_typed.json
│   └── nyc_stratified_s42.json
├── primitive_oracle/            # Circularity check
│   └── primitive_oracle_results.json
├── robustness/                  # Robustness experiments
│   ├── robustness_results.json
│   └── bank_scaling_results.json
├── cross_city/                  # Cross-city transfer
│   ├── nyc2tokyo_s42.json
│   └── tokyo2nyc_s42.json
├── case_study/                  # Qualitative analysis
│   └── porto_case_study.json
├── truezs/                      # True zero-shot protocol (v7)
│   ├── synthetic_seed{42,123,456}.json
│   └── porto_seed{42,123,456}.json
├── diagnostic_plots/            # Visualization figures (PDF + PNG)
│   ├── confusion_matrix_{synthetic,porto}.{pdf,png}
│   ├── error_rejection_{synthetic,porto}.{pdf,png}
│   └── zs_scores_{synthetic,porto}.{pdf,png}
└── older_versions/v5/           # Earlier experiment versions
```

---

## Offline Server Usage

For GPU servers without internet access:

```bash
# 1. On a machine with internet, download models
python scripts/download_for_offline.py --output models_cache/

# 2. Copy models_cache/ to the server

# 3. On the server, set environment variables
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 4. Train with local model path
python -m langtraj_osr.train --use_synthetic --seed 42 \
    --text_encoder models_cache/all-MiniLM-L6-v2 \
    --output_dir results/synthetic/seed_42
```

---

## Citation

```bibtex
@inproceedings{langtraj2026kdd,
  title={LangTraj-OSR: Language-Guided Open-Set Anomaly Detection
         and Concept Assignment in Semantic Trajectories},
  author={Anonymous},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on
             Knowledge Discovery and Data Mining},
  year={2026}
}
```

## License

This code is released for academic research purposes.
