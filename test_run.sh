#!/usr/bin/env bash
set -euo pipefail

# Quick test runner that combines --trial_run and --test_run for semlaflow training.
# Usage: ./test_run.sh [qm9|geom-drugs] [extra argparse flags...]
# Runs from the VEDA repository root so relative --data_path values resolve.

usage() {
  cat <<'EOF'
Usage: ./test_run.sh [dataset] [extra args...]

Datasets:
  qm9         -> ../semla-flow/data/qm9/smol/, val every 10 epochs, bond loss 0.5, warm-up 2000
  geom-drugs  -> ../semla-flow/data/geom-drugs/smol/, val every 5 epochs

Any additional arguments are passed straight through to python -m semlaflow.train.
EOF
}

DATASET="${1:-qm9}"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

case "$DATASET" in
  qm9)
    DATA_PATH="../semla-flow/data/qm9/smol/"
    EPOCHS=300
    VAL_CHECK=10
    EXTRA_ARGS=(--bond_loss_weight 0.5 --warm_up_steps 2000)
    ;;
  geom-drugs)
    DATA_PATH="../semla-flow/data/geom-drugs/smol/"
    EPOCHS=300
    VAL_CHECK=5
    EXTRA_ARGS=()
    ;;
  *)
    echo "Unknown dataset '$DATASET'" >&2
    usage
    exit 1
    ;;
esac

CMD=(
  python train.py
  --data_path "$DATA_PATH"
  --dataset "$DATASET"
  --epochs "$EPOCHS"
  --mask_rate_strategy edm
  --optimal_transport None
  --use_cat_time_based_weight
  --val_check_epochs "$VAL_CHECK"
  "${EXTRA_ARGS[@]}"
  "$@"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}"

