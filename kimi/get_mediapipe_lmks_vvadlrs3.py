import os
from glob import glob
import numpy as np
from pathlib import Path
import h5py
import csv

import get_mediapipe_lmks as gml

# Iterate through all of files in all of the folders in the dataset
source_main_path = "/home/ksw38/.cache/kagglehub/datasets/adrianlubitz/vvadlrs3/versions/4/faceImages_small.h5"
out_main_path = "/home/ksw38/groups/grp_landmarks/nobackup/autodelete/landmarks_vvadlrs3_mp/main"

def save_landmarks(data, y, name):
    folder_path = os.path.join(out_main_path, name)
    os.makedirs(folder_path, exist_ok=True)
    num_skipped_files = 0
    skipped_files = []
    index_file_pairs = []
    for i, frames in enumerate(data):
        try:
            video_lmks = []
            for frame in frames:
                frame = frame.copy()
                lmks = gml.get_lmks(frame)
                if lmks is None:
                    raise(TypeError("Lmks is None"))
                lmks_np = np.array([[lm.x, lm.y, lm.z] for lm in lmks])
                video_lmks.append(lmks_np)
            video_lmks_np = np.stack(video_lmks)

            file_name = os.path.join(folder_path, str(i) + ".npy")

            np.save(file_name, video_lmks_np)
            index_file_pairs.append((i, file_name, y[i]))
        
        except Exception as e:
            print(f"Error processing {i}: {e}")
            num_skipped_files += 1
            skipped_files.append(i)
            continue

    csv_path = os.path.join(out_main_path, f"{name}_files.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "file_path", "y"])
        writer.writerows(index_file_pairs)

    print(f"Saved {len(index_file_pairs)} files. Skipped {num_skipped_files}.")
    print(f"CSV written to {csv_path}")

    if skipped_files:
        print(f"Skipped indices: {skipped_files}")
            


with h5py.File(source_main_path, 'r') as f:
    # Get frames from the h5 file
    x_test = f['x_test']
    x_train = f['x_train']
    # Get the ground truth labels
    y_test = f['y_test']
    y_train = f['y_train']

    save_landmarks(x_test, y_test, "x_test")
    save_landmarks(x_train, y_train, "x_train")
    
