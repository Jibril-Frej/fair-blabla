#!/usr/bin/env bash
# The French exercise experiment: Qwen and Mistral each predict, then each judges the other's predictions.
# Usage (repo root on the login node): bash slurm/exercises_all.sh
set -euo pipefail
mkdir -p logs
QWEN=qwen3.8-27b-fp8-structured
MISTRAL=mistral-small-3.2-24b-structured

# Mistral Small 3.2 ships its tokenizer in Mistral's format only (tekken.json); no images are used.
# slurm/compat works around a vLLM 0.30 / transformers 5.17 import error in vLLM's Pixtral module.
mistral() {  # steps [sbatch args]
  SERVER_PYTHONPATH=slurm/compat MODEL_ID=mistralai/Mistral-Small-3.2-24B-Instruct-2506 SLUG=$MISTRAL STEPS="$1" \
    VLLM_ARGS='--tokenizer-mode mistral --config-format mistral --load-format mistral --limit-mm-per-prompt {"image":0}' \
    sbatch --parsable --export=ALL "${@:2}" slurm/exercises.sbatch
}
# The Qwen3.8 checkpoints include a vision encoder; no images are used.
qwen() {
  MODEL_ID=Qwen/Qwen3.8-27B-FP8 SLUG=$QWEN NO_THINKING=1 STEPS="$1" \
    VLLM_ARGS='--limit-mm-per-prompt {"image":0,"video":0}' \
    sbatch --parsable --export=ALL "${@:2}" slurm/exercises.sbatch
}

PM=$(mistral predict); echo "$MISTRAL predict: $PM"
PQ=$(qwen predict); echo "$QWEN predict: $PQ"
echo "$MISTRAL judges $QWEN: $(mistral "judge:$QWEN" --dependency=afterok:"$PQ")"
echo "$QWEN judges $MISTRAL: $(qwen "judge:$MISTRAL" --dependency=afterok:"$PM")"
