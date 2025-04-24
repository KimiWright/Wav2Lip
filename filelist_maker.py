import os
import random
from glob import glob
from os import path

# Define paths
data_root = "/home/ksw38/groups/grp_lip/nobackup/autodelete/datasets/fslgroup/grp_lip/compute/datasets/LRS2/preprocessedRetinaface/lrs2/lrs2_video_seg24s/mvlrs_v1/main"#"mvlrs_v1/main" 
filelists_dir = "filelists"

# Create the output directory if it doesn't exist
os.makedirs(filelists_dir, exist_ok=True)

# Get all video files (remove extensions)
# video_files = sorted([f[:-4] for f in os.listdir(data_root) if f.endswith(".mp4")])
video_files = glob(path.join(data_root, '*/*.mp4'))

# Shuffle data for randomness
random.seed(42)
random.shuffle(video_files)

# Split data (adjust ratios if needed)
train_ratio, val_ratio = 0.8, 0.1  # 80% train, 10% val, 10% test
train_end = int(len(video_files) * train_ratio)
val_end = train_end + int(len(video_files) * val_ratio)

train_list = video_files[:train_end]
val_list = video_files[train_end:val_end]
test_list = video_files[val_end:]

# Save filelists
for name, data in zip(["train", "val", "test"], [train_list, val_list, test_list]):
    with open(os.path.join(filelists_dir, f"{name}.txt"), "w") as f:
        f.write("\n".join(data))

print("Filelists generated in 'filelists/' directory.")
