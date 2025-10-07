#!/bin/bash --login

#SBATCH --time=20:00:00   # walltime
#SBATCH --gpus=1
#SBATCH --mem=100G   # memory per CPU core
#SBATCH --mail-user=KimiWright64@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --cpus-per-task=1
#SBATCH --chdir /home/ksw38/RVL/color_syncnet/Wav2Lip/kimi/

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate vsr
python /home/ksw38/RVL/color_syncnet/Wav2Lip/kimi/get_landmarks.py