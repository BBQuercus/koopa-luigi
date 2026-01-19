#!/bin/bash

#SBATCH --account=fly-image
#SBATCH --cpus-per-task=48   # CPU cores per task
#SBATCH --job-name="koopa-cpu"
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=128G  # set memory requirements
#SBATCH --nodes=1  # ensure that all cores are on one machine
#SBATCH --ntasks=1  # non MPI applications are usually single task
#SBATCH --output=./logs/%x-%j.log
#SBATCH --partition=cpu_short  # partition (cpu_short for jobs <12:00:00)
#SBATCH --time=12:00:00  # time required by the job

# Configuration
CONFIG=PATH/TO/KOOPA.cfg
WORKERS=8

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Try legacy environment first (works with most existing models)
echo "Attempting with LEGACY environment (TF 2.13)..."

if nice -n19 "$SCRIPT_DIR/run_legacy.sh" --config $CONFIG --workers $WORKERS; then
    echo "Pipeline completed successfully with legacy environment."
    exit 0
fi

# Legacy failed - try modern environment
echo ""
echo "Legacy environment failed. Trying MODERN environment (TF 2.17+)..."

if nice -n19 "$SCRIPT_DIR/run_modern.sh" --config $CONFIG --workers $WORKERS; then
    echo "Pipeline completed successfully with modern environment."
    exit 0
fi

# Both failed
echo ""
echo "ERROR: Pipeline failed with both environments."
exit 1
