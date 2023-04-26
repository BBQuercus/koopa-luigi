#!/bin/sh

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
CONDA_DIR=/tungstenfs/scratch/gchao/eichbast/miniconda/bin/activate
CONFIG=PATH/TO/KOOPA.cfg

# Run
source $CONDA_DIR koopa
koopa-luigi --config $CONFIG --gpu
