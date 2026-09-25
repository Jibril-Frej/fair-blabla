#!/usr/bin/env bash
# Pull the vLLM OpenAI server image (pinned by digest) into ~/fair-blabla-cluster/.
# The pull is memory-heavy, so it runs as a CPU Slurm job, not on the login node.
# Usage (on the login node): sbatch slurm/fetch_vllm_image.sh
#SBATCH --job-name=fb-fetch-vllm
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
# vllm/vllm-openai:v0.30.0, resolved 2026-09-25
DIGEST="sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90"
DEST="$HOME/fair-blabla-cluster"
mkdir -p "$DEST" "/scratch/$USER/aptmp" "/scratch/$USER/apcache"
export APPTAINER_TMPDIR="/scratch/$USER/aptmp" APPTAINER_CACHEDIR="/scratch/$USER/apcache"
apptainer pull --force "$DEST/vllm-openai.sif" "docker://vllm/vllm-openai@$DIGEST"
ls -lh "$DEST/vllm-openai.sif"
rm -rf "/scratch/$USER/aptmp" "/scratch/$USER/apcache"
