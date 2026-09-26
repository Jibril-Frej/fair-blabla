#!/usr/bin/env bash
# Submit the whole evaluation (repo root on the login node): retrieval, then one job per model.
# Each model gets one H200 (a 27B model in BF16 fits in 141 GB); the QoS allows 2 H200 at a time,
# so Slurm runs the model jobs two by two.
# Usage: bash slurm/submit_all.sh [slug ...]   (default: every model below)
set -euo pipefail
mkdir -p logs
QWEN_METHODS=${QWEN_METHODS:-"linguistic_indicators demographic_axes guardian_criteria biasalert_rag"}
# The Qwen3.8 checkpoints include a vision encoder; no images are used.
QWEN_ARGS='--limit-mm-per-prompt {"image":0,"video":0}'

# Retrieval runs once; its output is reused when it is already there.
DEP=()
if [ ! -s results/retrieval/demographic_axes.jsonl ] || [ ! -s results/retrieval/biasalert.jsonl ]; then
  RETRIEVE=$(sbatch --parsable slurm/retrieve.sbatch)
  echo "retrieval: $RETRIEVE"
  DEP=(--dependency "afterok:$RETRIEVE")
fi

submit() {  # slug model_id display precision kind methods max_len vllm_args [structured]
  local id
  id=$(SLUG=$1 MODEL_ID=$2 DISPLAY_NAME=$3 PRECISION=$4 KIND=$5 METHODS=$6 MAX_LEN=$7 VLLM_ARGS=$8 STRUCTURED=${9:-} \
    sbatch --parsable --export=ALL --gres=gpu:h200:1 --job-name "fb-$1" "${DEP[@]}" slurm/serve_and_run.sbatch)
  echo "$1: $id"
}

want() { [ ${#SLUGS[@]} -eq 0 ] || [[ " ${SLUGS[*]} " == *" $1 "* ]]; }
SLUGS=("$@")
want granite-guardian-3.3-8b && submit granite-guardian-3.3-8b ibm-granite/granite-guardian-3.3-8b \
  "granite-guardian-3.3-8b (BF16)" BF16 granite "guardian_criteria guardian_social_bias" 8192 ""
want qwen3.8-27b-bf16 && submit qwen3.8-27b-bf16 Qwen/Qwen3.8-27B "Qwen3.8-27B (BF16)" BF16 chat "$QWEN_METHODS" 16384 "$QWEN_ARGS"
want qwen3.8-27b-fp8 && submit qwen3.8-27b-fp8 Qwen/Qwen3.8-27B-FP8 "Qwen3.8-27B (FP8)" FP8 chat "$QWEN_METHODS" 16384 "$QWEN_ARGS"
want qwen3.8-27b-uncensored-fp8 && submit qwen3.8-27b-uncensored-fp8 orcarouter/Qwen3.8-27B-Uncensored-FP8 \
  "Qwen3.8-27B-Uncensored (FP8)" FP8 chat "$QWEN_METHODS" 16384 "$QWEN_ARGS"
# Same models with constrained answers (vLLM structured outputs)
want qwen3.8-27b-bf16-structured && submit qwen3.8-27b-bf16-structured Qwen/Qwen3.8-27B \
  "Qwen3.8-27B (BF16), structured" BF16 chat "$QWEN_METHODS" 16384 "$QWEN_ARGS" 1
want qwen3.8-27b-fp8-structured && submit qwen3.8-27b-fp8-structured Qwen/Qwen3.8-27B-FP8 \
  "Qwen3.8-27B (FP8), structured" FP8 chat "$QWEN_METHODS" 16384 "$QWEN_ARGS" 1
want qwen3.8-27b-uncensored-fp8-structured && submit qwen3.8-27b-uncensored-fp8-structured \
  orcarouter/Qwen3.8-27B-Uncensored-FP8 "Qwen3.8-27B-Uncensored (FP8), structured" FP8 chat "$QWEN_METHODS" 16384 "$QWEN_ARGS" 1
exit 0
