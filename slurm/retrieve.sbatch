#!/usr/bin/env bash
# Retrieve the in-context examples (BGE-M3) and BiasAlert references (contriever-msmarco)
# for all evaluation items -> results/retrieval/. Runs inside the vLLM image (torch + transformers).
# Usage (repo root on the login node): sbatch slurm/retrieve.sbatch
#SBATCH --job-name=fb-retrieve
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
SIF="$HOME/fair-blabla-cluster/vllm-openai.sif"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
export HF_HUB_OFFLINE=1
apptainer exec --nv --env PYTHONPATH=src "$SIF" python3 scripts/retrieve.py
