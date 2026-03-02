#!/usr/bin/env bash
set -e

# Simple wrapper around the Python entry points in src/.
# Example:
#   bash run_experiment.sh --mode robustness --data data/test --backends pytesseract,easyocr --out results

MODE="robustness"
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

case "$MODE" in
  robustness)
    python src/run_robustness.py "${ARGS[@]}"
    ;;
  finetune)
    # Optional: small CRNN finetune (GPU-only), mainly driven via notebook 03.
    # This entry point is kept minimal and delegates to finetune_crnn.py.
    python src/finetune_crnn.py "${ARGS[@]}"
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Supported modes: robustness, finetune"
    exit 1
    ;;
esac

