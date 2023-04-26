#!/bin/sh

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
CONDA_DIR=/tungstenfs/scratch/gchao/eichbast/miniconda/bin/activate
CONFIG=PATH/TO/KOOPA.cfg

# Run
source $CONDA_DIR koopa
koopa-luigi --config $CONFIG --workers 8
