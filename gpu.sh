#!/bin/bash

#SBATCH --account=fly-image
#SBATCH --cpus-per-task=16   # CPU cores per task
#SBATCH --gres=gpu:a40:1  # what kind and how many GPUs do you need
#SBATCH --job-name="koopa-gpu"
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=64G  # set memory requirements
#SBATCH --nodes=1  # ensure that all cores are on one machine
#SBATCH --ntasks=1  # non MPI applications are usually single task
#SBATCH --output=./logs/%x-%j.log
#SBATCH --partition=gpu_short  # partition (gpu_short for jobs <12:00:00)
#SBATCH --time=04:00:00  # time required by the job, if you hit this limit the job will be terminated

# Configuration
CONFIG=PATH/TO/KOOPA.cfg

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Try legacy environment first (works with most existing models)
echo "Attempting with LEGACY environment (TF 2.13)..."

if "$SCRIPT_DIR/run_legacy.sh" --config $CONFIG --gpu; then
    echo "Pipeline completed successfully with legacy environment."
    exit 0
fi

# Legacy failed - try modern environment
echo ""
echo "Legacy environment failed. Trying MODERN environment (TF 2.17+)..."

if "$SCRIPT_DIR/run_modern.sh" --config $CONFIG --gpu; then
    echo "Pipeline completed successfully with modern environment."
    exit 0
fi

# Both failed
echo ""
echo "ERROR: Pipeline failed with both environments."
exit 1
