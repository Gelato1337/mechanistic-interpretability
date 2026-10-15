#!/bin/bash
# Example: the local LLM judge (thesis/analysis/classes/judge_vllm.py, Qwen3.5-122B-A10B through vLLM) on one LUMI GPU node.
# Usage: VLLM_SIF=/path/to/vllm-rocm.sif sbatch --account=<project> slurm/judge_lumi.sh
#        CARD_DIRS=work/dom_cards2 OUT=work/qwen_dom2_labels.jsonl PROTOCOL=thesis/analysis/classes/judge/protocol_qwen_v6.txt sbatch ...
#SBATCH --partition=small-g
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=56 --gpus-per-node=8
#SBATCH --mem=480G --time=06:00:00
#SBATCH --output=logs/judge-%j.out
set -euo pipefail
: "${VLLM_SIF:?set VLLM_SIF to a vLLM ROCm container image}"
export SAE_WORK=${SAE_WORK:-$PWD/work}
export CARD_DIRS="${CARD_DIRS:-${SAE_WORK}/pilot_cards:${SAE_WORK}/pilot_cards_v3}"
export OUT="${OUT:-${SAE_WORK}/qwen_labels.jsonl}"
export PROTOCOL="${PROTOCOL:-$PWD/thesis/analysis/classes/judge/protocol_cards.txt}"
export LIMIT="${LIMIT:-0}" THINK="${THINK:-0}"
export HF_HOME=${HF_HOME:-$PWD/hf_cache}          # the model weights must already be in this cache
export HF_HUB_OFFLINE=1
# vLLM on MI250X: settings that worked for the thesis run
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export VLLM_WORKER_MULTIPROC_METHOD=spawn NCCL_NET=Socket NUMEXPR_MAX_THREADS=16
export VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_AITER_MHA=0 VLLM_ROCM_USE_AITER_RMSNORM=0 VLLM_ROCM_USE_AITER_MLA=0
export VLLM_USE_TRITON_FLASH_ATTN=1 VLLM_MLA_DISABLE=1
export PYTHONNOUSERSITE=1; unset PYTHONPATH
# vLLM's ZeroMQ sockets need a short TMPDIR path
export TMPDIR=/tmp/judge_${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
srun singularity exec --rocm -B /pfs,/scratch,/flash,/appl,"$TMPDIR" "$VLLM_SIF" python3 -u thesis/analysis/classes/judge_vllm.py
