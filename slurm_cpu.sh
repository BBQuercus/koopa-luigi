#!/bin/sh

#SBATCH --mail-user=
#SBATCH --mail-type=END
#SBATCH --account=fly-image
#SBATCH --job-name="koopa-cpu" # Something that helps you recognize your job in the queue
#SBATCH --output=./logs/%x-%j.log
#SBATCH --nodes=1 # ensure that all cores are on one machine
#SBATCH --ntasks=1    # non MPI applications are usually single task, but you can do multiple tasks as well
#SBATCH --time=12:00:00    # time required by the job, if you hit this limit the job will be terminated
#SBATCH --cpus-per-task=28   # how many CPU cores per task do you need?
#SBATCH --partition=cpu_short # for CPU only change this, e.g. to cpu_short, see also sinfo command output

# Configuration
CONDA_DIR=/tungstenfs/scratch/gchao/eichbast/miniconda/bin/activate
CONFIG=/tungstenfs/scratch/gchao/eichbast/koopa-flows/tests/config/flies.cfg

# Run
source $CONDA_DIR koopa
koopa-luigi --config $CONFIG --workers 28
