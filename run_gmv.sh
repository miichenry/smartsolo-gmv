#!/bin/bash
#SBATCH --job-name=gmv
#SBATCH --partition=public-interactive-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=output/gmv_%j.log
#SBATCH --error=output/gmv_%j.log

# Usage:
#   sbatch run_gmv.sh config/run_indonesia.yaml gif
#   sbatch run_gmv.sh config/run_alaska.yaml mp4
#   sbatch run_gmv.sh config/run_newcaledonia.yaml gif

CONFIG=${1:?Usage: sbatch run_gmv.sh <config.yaml> [format]}
FORMAT=${2:-gif}

source /opt/ebsofts/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate obspy

echo "Running GMV: config=$CONFIG format=$FORMAT"
echo "Node: $(hostname), Date: $(date)"
echo "---"

python gmv.py "$CONFIG" --format "$FORMAT"

echo "---"
echo "Finished: $(date)"
