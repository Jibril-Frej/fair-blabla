#!/usr/bin/env bash
# Download the model weights into the Hugging Face cache (~/.cache/huggingface).
# orcarouter/Qwen3.8-27B-Uncensored-FP8 is gated: accept its terms on huggingface.co
# and log in on the cluster (`uvx hf auth login`) before running this.
# Usage: sbatch slurm/download_models.sbatch [model ...]
#SBATCH --job-name=fb-download
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
export PATH="/software/bin:$HOME/.local/bin:$PATH"
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(
    Qwen/Qwen3.8-27B
    Qwen/Qwen3.8-27B-FP8
    orcarouter/Qwen3.8-27B-Uncensored-FP8
    ibm-granite/granite-guardian-3.3-8b
    BAAI/bge-m3
    facebook/contriever-msmarco
  )
fi
failed=()
for m in "${MODELS[@]}"; do
  echo "=== $m"
  uvx --from huggingface_hub hf download "$m" \
    --exclude "*.gguf" --exclude "*.pth" --exclude "*.onnx*" --exclude "onnx/*" || failed+=("$m")
done
echo "failed: ${failed[*]:-none}"
