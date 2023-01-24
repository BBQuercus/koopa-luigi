#!/bin/sh

#SBATCH --mail-user=
#SBATCH --mail-type=END
#SBATCH --account=fly-image
#SBATCH --job-name="koopa-gpu" # Something that helps you recognize your job in the queue
#SBATCH --output=./logs/%x-%j.log
#SBATCH --nodes=1 # ensure that all cores are on one machine
#SBATCH --ntasks=1    # non MPI applications are usually single task, but you can do multiple tasks as well
#SBATCH --time=04:00:00    # time required by the job, if you hit this limit the job will be terminated
#SBATCH --cpus-per-task=8   # how many CPU cores per task do you need?
#SBATCH --partition=gpu_short # for CPU only change this, e.g. to cpu_short, see also sinfo command output
#SBATCH --gres=gpu:a40:1  # what kind and how many GPUs do you need. We only have v100, max 4 per node

# Configuration
CONDA_DIR=/tungstenfs/scratch/gchao/eichbast/miniconda/bin/activate
CONFIG=PATH/TO/KOOPA.cfg

# Run
source $CONDA_DIR koopa
koopa-luigi --config $CONFIG --gpu
