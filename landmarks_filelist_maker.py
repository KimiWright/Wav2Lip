import os
import random
from glob import glob
from os import path
from collections import defaultdict

# Define paths
data_root = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main/" 
filelists_dir = "landmarks_filelists"

# Create the output directory if it doesn't exist
os.makedirs(filelists_dir, exist_ok=True)

lmk_files = glob(path.join(data_root, '*/*.npy'))

## This groups all files with the same number code, regardlss of folder, which is not nessarily correct, 
## but is likely okay for the purposes of this code.
# Group files by the first 5 characters of their base filename
grouped_files = defaultdict(list)
for file in lmk_files:
    basename = path.basename(file)
    folder = path.basename(path.dirname(file))
    group_key = f"{folder}_{basename[:5]}"  # First 5 characters
    grouped_files[group_key].append(file)

# Convert grouped values to list
groups = list(grouped_files.values())

# Shuffle groups instead of individual files
random.seed(42)
random.shuffle(groups)

# Flatten grouped files after splitting
num_groups = len(groups)
train_ratio, val_ratio = 0.8, 0.1

train_end = int(num_groups * train_ratio)
val_end = train_end + int(num_groups * val_ratio)

train_list = [f for group in groups[:train_end] for f in group]
val_list   = [f for group in groups[train_end:val_end] for f in group]
test_list  = [f for group in groups[val_end:] for f in group]

# Save filelists
for name, data in zip(["train", "val", "test"], [train_list, val_list, test_list]):
    with open(os.path.join(filelists_dir, f"{name}.txt"), "w") as f:
        f.write("\n".join(data))

print(f"Filelists generated in {filelists_dir} directory.")
