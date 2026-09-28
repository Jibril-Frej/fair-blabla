#!/usr/bin/env bash
# The French exercise experiment: Qwen and Mistral each predict, and each judges the other's topics.
# Qwen's predictions and topic similarities already exist, so its job only judges Mistral's topics.
# Usage: bash slurm/exercises_all.sh
# Run from the repo root on the login node: bash slurm/exercises_all.sh
set -euo pipefail
mkdir -p logs
QWEN=qwen3.8-27b-fp8-structured
MISTRAL=mistral-small-3.2-24b-structured

# Mistral Small 3.2 ships its tokenizer in Mistral's format only (tekken.json); no images are used.
# slurm/compat works around a vLLM 0.30 / transformers 5.17 import error in vLLM's Pixtral module.
JOB=$(SERVER_PYTHONPATH=slurm/compat MODEL_ID=mistralai/Mistral-Small-3.2-24B-Instruct-2506 SLUG=$MISTRAL STEPS="predict judge:$QWEN embed" \
  VLLM_ARGS='--tokenizer-mode mistral --config-format mistral --load-format mistral --limit-mm-per-prompt {"image":0}' \
  sbatch --parsable --export=ALL slurm/exercises.sbatch)
echo "$MISTRAL: $JOB"

# The Qwen3.8 checkpoints include a vision encoder; no images are used.
MODEL_ID=Qwen/Qwen3.8-27B-FP8 SLUG=$QWEN NO_THINKING=1 STEPS="judge:$MISTRAL" \
  VLLM_ARGS='--limit-mm-per-prompt {"image":0,"video":0}' \
  sbatch --parsable --export=ALL --dependency=afterok:"$JOB" slurm/exercises.sbatch
