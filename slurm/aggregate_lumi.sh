#!/bin/bash
# Example: per-domain statistics from a capture (CPU node).  Usage: sbatch --account=<project> slurm/aggregate_lumi.sh 27b
#SBATCH --partition=small --cpus-per-task=8 --mem=128G --time=06:00:00
#SBATCH --output=logs/aggregate-%j.out
set -euo pipefail
MODEL=${1:-27b}
python scripts/aggregate.py --capture runs/${MODEL}_all --prompts thesis/data/prompts.csv --out stats/262k/${MODEL}
