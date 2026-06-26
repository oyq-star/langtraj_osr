#!/bin/bash
# Run all experiment phases in order for LangTraj-OSR
# Usage: bash langtraj_osr/scripts/run_all.sh [GPU_ID] [--synthetic] [--text_encoder PATH]
#
# Follows the experiment plan:
#   Phase 1: Benchmark construction
#   Phase 2: Sanity baselines (NUMOSIM)          — gate: NormOnly < 90% AUROC
#   Phase 3: Full model (NUMOSIM + GeoLife)
#   Phase 4: Zero-shot composition               — gate: Language > DSL-XL
#   Phase 5: Zero-shot family
#   Phase 6: Full baseline suite (all datasets)   — gate: Full > ATROM
#   Phase 7: Cross-city transfer
#   Phase 8: Robustness studies
#   Phase 9: All ablations

set -e

GPU=${1:-0}
SYNTHETIC_FLAG=""
TEXT_ENCODER_FLAG=""

shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --synthetic) SYNTHETIC_FLAG="--use_synthetic"; shift ;;
        --text_encoder) TEXT_ENCODER_FLAG="--text_encoder $2"; shift 2 ;;
        *) shift ;;
    esac
done

OUTPUT_DIR="results"
SEED=42

echo "========================================"
echo "LangTraj-OSR Full Experiment Suite"
echo "GPU: $GPU | Seed: $SEED"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Phase 1: Benchmark Construction
echo ""
echo "[Phase 1/9] Benchmark Construction"
python -m langtraj_osr.scripts.run_phase --phase 1 --gpu $GPU --seed $SEED --output_dir $OUTPUT_DIR $SYNTHETIC_FLAG $TEXT_ENCODER_FLAG
echo "Phase 1 complete."

# Phase 2: Sanity Baselines
echo ""
echo "[Phase 2/9] Sanity Baselines (NUMOSIM)"
python -m langtraj_osr.scripts.run_phase --phase 2 --gpu $GPU --seed $SEED --output_dir $OUTPUT_DIR $SYNTHETIC_FLAG $TEXT_ENCODER_FLAG
echo "Phase 2 complete. Check go/no-go gate before proceeding."

# Phase 3: Full Model
echo ""
echo "[Phase 3/9] Full Model (NUMOSIM + GeoLife)"
python -m langtraj_osr.scripts.run_phase --phase 3 --gpu $GPU --seed $SEED --output_dir $OUTPUT_DIR $SYNTHETIC_FLAG $TEXT_ENCODER_FLAG
echo "Phase 3 complete."

# Phase 4: Zero-Shot Composition
echo ""
echo "[Phase 4/9] Zero-Shot Composition Split"
python -m langtraj_osr.scripts.run_phase --phase 4 --gpu $GPU --seed $SEED --output_dir $OUTPUT_DIR $SYNTHETIC_FLAG $TEXT_ENCODER_FLAG
echo "Phase 4 complete. Check go/no-go gate."

# Phase 5: Zero-Shot Family
echo ""
echo "[Phase 5/9] Zero-Shot Family Split"
python -m langtraj_osr.scripts.run_phase --phase 5 --gpu $GPU --seed $SEED --output_dir $OUTPUT_DIR $SYNTHETIC_FLAG $TEXT_ENCODER_FLAG
echo "Phase 5 complete."

# Phase 6: Full Baseline Suite
echo ""
echo "[Phase 6/9] Full Baseline Suite (all datasets)"
python -m langtraj_osr.scripts.run_phase --phase 6 --gpu $GPU --seed $SEED --output_dir $OUTPUT_DIR $SYNTHETIC_FLAG $TEXT_ENCODER_FLAG
echo "Phase 6 complete. Check go/no-go gate."

# Phase 7: Cross-City Transfer
echo ""
echo "[Phase 7/9] Cross-City Transfer (NYC <-> Tokyo)"
python -m langtraj_osr.scripts.run_phase --phase 7 --gpu $GPU --seed $SEED --output_dir $OUTPUT_DIR $SYNTHETIC_FLAG $TEXT_ENCODER_FLAG
echo "Phase 7 complete."

# Phase 8: Robustness Studies
echo ""
echo "[Phase 8/9] Robustness (Paraphrase + Analyst Noise)"
python -m langtraj_osr.scripts.run_phase --phase 8 --gpu $GPU --seed $SEED --output_dir $OUTPUT_DIR $SYNTHETIC_FLAG $TEXT_ENCODER_FLAG
echo "Phase 8 complete."

# Phase 9: Ablations
echo ""
echo "[Phase 9/9] All Ablations (A1-A11)"
python -m langtraj_osr.scripts.run_phase --phase 9 --gpu $GPU --seed $SEED --output_dir $OUTPUT_DIR $SYNTHETIC_FLAG $TEXT_ENCODER_FLAG
echo "Phase 9 complete."

echo ""
echo "========================================"
echo "All phases complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Review results in $OUTPUT_DIR/"
echo "  2. Run: /auto-review-loop 'LangTraj-OSR'"
echo "  3. Run: /paper-writing"
echo "========================================"
