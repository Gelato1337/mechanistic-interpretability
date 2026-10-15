#!/bin/bash
# Example: capture all layers of one Gemma 3 size on one LUMI GPU node (8 x MI250X GCD).
# Usage: sbatch --account=<project> slurm/capture_lumi.sh 27b
#SBATCH --partition=standard-g
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 --gpus-per-node=8
#SBATCH --mem=480G --time=02:00:00
#SBATCH --output=logs/capture-%j.out
set -euo pipefail
MODEL=${1:-27b}
# The LUMI PyTorch container ships a Triton that compiles the fused kernel; the CSC pytorch module's Triton did not.
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif
export HF_HOME=${HF_HOME:-$PWD/hf_cache}          # needs access to google/gemma-3-* (gated) and google/gemma-scope-2-*
export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN}
export TRITON_CACHE_DIR=$PWD/.triton_cache
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
srun singularity exec -B /pfs,/scratch,/flash,/appl $SIF bash -c "\$WITH_CONDA; source venv/bin/activate; \
  python scripts/capture.py --model $MODEL --prompts thesis/data/prompts.csv --out runs/${MODEL}_all --sae-bf16-dir sae_bf16"
