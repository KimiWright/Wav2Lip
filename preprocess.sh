#!/bin/bash --login

#SBATCH --time=10:00:00   # walltime
#SBATCH --gpus=1
#SBATCH --mem=100G   # memory per CPU core
#SBATCH --mail-user=KimiWright64@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --cpus-per-task=1
#SBATCH --chdir /home/ksw38/RVL/color_syncnet/Wav2Lip

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate color_syncnet
python /home/ksw38/RVL/color_syncnet/Wav2Lip/preprocess.py --data_root /home/ksw38/groups/grp_lip/nobackup/autodelete/datasets/fslgroup/grp_lip/compute/datasets/LRS2/preprocessedRetinaface/lrs2/lrs2_video_seg24s/mvlrs_v1/main --preprocessed_root /home/ksw38/RVL/color_syncnet/Wav2Lip/lrs2_preprocessed/