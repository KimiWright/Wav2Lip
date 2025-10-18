#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32G
#SBATCH --array=0-199
#SBATCH -o /home/ksw38/RVL/color_syncnet/Wav2Lip/kimi/slurm/%A_%x_%a.out
#SBATCH -e /home/ksw38/RVL/color_syncnet/Wav2Lip/kimi/slurm/Error_%A_%x_%a.out
#SBATCH --chdir /home/ksw38/RVL/color_syncnet/Wav2Lip/kimi/
#SBATCH -J genLmks%a


# Run the worker with its array index and total groups
python /home/ksw38/RVL/color_syncnet/Wav2Lip/kimi/lmkGen.py \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --groups 200 \
  --exclude-train
