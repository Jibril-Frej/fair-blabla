#!/usr/bin/env bash
# Start a vLLM server for one model inside the job, run the bias detection methods against it,
# then stop the server. Results go to results/$SLUG/.
# Settings come from the environment, e.g. (repo root on the login node):
#   sbatch --gres=gpu:h200:1 --export=ALL,MODEL_ID=Qwen/Qwen3.8-27B,SLUG=qwen3.8-27b-bf16,\
#     DISPLAY_NAME="Qwen3.8-27B (BF16)",PRECISION=BF16,KIND=chat,\
#     METHODS="linguistic_indicators demographic_axes guardian_criteria biasalert_rag" slurm/serve_and_run.sbatch
# Optional: TP (tensor parallel size, default 1), MAX_LEN (default 16384), VLLM_ARGS (extra server flags),
#   STRUCTURED=1 (constrained answers, see scripts/run_methods.py --structured).
#SBATCH --job-name=fb-run
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${MODEL_ID:?}" "${SLUG:?}" "${DISPLAY_NAME:?}" "${METHODS:?}"
KIND="${KIND:-chat}" TP="${TP:-1}" MAX_LEN="${MAX_LEN:-16384}" PRECISION="${PRECISION:-}"
SIF="$HOME/fair-blabla-cluster/vllm-openai.sif"
PORT=$((8100 + SLURM_JOB_ID % 800))
export HF_HUB_OFFLINE=1
echo "node=$(hostname) model=$MODEL_ID start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

apptainer exec --nv "$SIF" vllm serve "$MODEL_ID" --served-model-name model \
  --host 127.0.0.1 --port "$PORT" --tensor-parallel-size "$TP" --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.90 --seed 0 ${VLLM_ARGS:-} > "logs/vllm-$SLUG-$SLURM_JOB_ID.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null || true' EXIT

for i in $(seq 1 180); do
  if ! kill -0 $SRV 2>/dev/null; then echo "server died"; tail -40 "logs/vllm-$SLUG-$SLURM_JOB_ID.log"; exit 1; fi
  if curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" > /dev/null; then echo "server ready after $((i * 10))s"; break; fi
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "server not ready"; exit 1; }
grep -m1 -i "vllm api server version\|version" "logs/vllm-$SLUG-$SLURM_JOB_ID.log" || true

EXTRA=()
[ "$KIND" = chat ] && EXTRA+=(--no-thinking)
[ -n "${STRUCTURED:-}" ] && EXTRA+=(--structured)
# shellcheck disable=SC2086
apptainer exec "$SIF" python3 scripts/run_methods.py --base-url "http://127.0.0.1:$PORT/v1" --model model \
  --slug "$SLUG" --display "$DISPLAY_NAME" --model-id "$MODEL_ID" --precision "$PRECISION" --kind "$KIND" \
  --methods $METHODS "${EXTRA[@]}"
echo "end=$(date -Is)"
