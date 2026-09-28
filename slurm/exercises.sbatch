#!/usr/bin/env bash
# Serve one model and run steps of the French exercise experiment with it (scripts/run_exercises.py):
#   predict           -> results/$SLUG/exercises/predictions.jsonl
#   judge:<slug>      -> judges the topics predicted by <slug>, results/<slug>/exercises/topic_judge.jsonl
#   embed             -> BGE-M3 topic similarities of $SLUG's predictions (after the server stops)
# Settings come from the environment, e.g. (repo root on the login node):
#   MODEL_ID=Qwen/Qwen3.8-27B-FP8 SLUG=qwen3.8-27b-fp8-structured NO_THINKING=1 STEPS="predict embed" \
#     VLLM_ARGS='--limit-mm-per-prompt {"image":0,"video":0}' sbatch --export=ALL slurm/exercises.sbatch
# SERVER_PYTHONPATH: extra PYTHONPATH for the vLLM server only (e.g. slurm/compat).
# See slurm/exercises_all.sh for the runs of the experiment.
#SBATCH --job-name=fb-exercises
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
: "${MODEL_ID:?}" "${SLUG:?}" "${STEPS:?}"
SIF="$HOME/fair-blabla-cluster/vllm-openai.sif"
PORT=$((8100 + SLURM_JOB_ID % 800))
export HF_HUB_OFFLINE=1
echo "node=$(hostname) model=$MODEL_ID steps=$STEPS start=$(date -Is)"

# shellcheck disable=SC2086
apptainer exec --nv ${SERVER_PYTHONPATH:+--env PYTHONPATH=$SERVER_PYTHONPATH} "$SIF" vllm serve "$MODEL_ID" --served-model-name model \
  --host 127.0.0.1 --port "$PORT" --max-model-len 16384 --gpu-memory-utilization 0.85 --seed 0 \
  ${VLLM_ARGS:-} > "logs/vllm-exercises-$SLUG-$SLURM_JOB_ID.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true' EXIT

for i in $(seq 1 180); do
  if ! kill -0 $SRV 2>/dev/null; then echo "server died"; tail -40 "logs/vllm-exercises-$SLUG-$SLURM_JOB_ID.log"; exit 1; fi
  if curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" > /dev/null; then echo "server ready after $((i * 10))s"; break; fi
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null || { echo "server not ready"; exit 1; }

EXTRA=()
[ -n "${NO_THINKING:-}" ] && EXTRA+=(--no-thinking)
run() {  # step [args]
  apptainer exec "$SIF" python3 scripts/run_exercises.py "$1" --base-url "http://127.0.0.1:$PORT/v1" --model model \
    --slug "$SLUG" "${EXTRA[@]}" "${@:2}"
}
for step in $STEPS; do
  case "$step" in
    predict) run predict ;;
    judge:*) run judge --predictions "${step#judge:}" ;;
    embed) ;;
    *) echo "unknown step $step"; exit 1 ;;
  esac
done

kill $SRV; wait $SRV 2>/dev/null || true
if [[ " $STEPS " == *" embed "* ]]; then
  apptainer exec --nv --env PYTHONPATH=src "$SIF" python3 scripts/embed_topics.py --slug "$SLUG"
fi
echo "end=$(date -Is)"
