import os
from glob import glob
import numpy as np
from pathlib import Path
import cv2
import h5py
from facetools import genMediapipeInfo,norm_lmks

# Iterate through all of files in all of the folders in the dataset
source_main_path = "/home/ksw38/.cache/kagglehub/datasets/adrianlubitz/vvadlrs3/versions/4/faceImages_small.h5"
out_main_path = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_vvadlrs3/main"
out_main_path = "/home/ksw38/groups/grp_landmarks/nobackup/autodelete/landmarks_vvadlrs3/main"

def save_landmarks(data, name):
    for i, frames in enumerate(data):
        try:
            # Make File Paths
            folder_path = os.path.join(out_main_path, name)
            os.makedirs(folder_path, exist_ok=True)

            file_name = str(i)
            out_path_lmks = os.path.join(out_main_path, name, file_name + "_lmks")
            out_path_yaw = os.path.join(out_main_path, name, file_name + "_yaw")
            out_path_pitch = os.path.join(out_main_path, name, file_name + "_pitch")
            out_path_roll = os.path.join(out_main_path, name, file_name + "_roll")
            
            print(f"Processing {file_name}...")
            # Generate and normalize landmarks
            
            _, lmks, allYaw, allPitch, allRoll = genMediapipeInfo(frames)
            lmks = norm_lmks(lmks)

            # Save the landmarks
            np.save(out_path_lmks, lmks)
            np.save(out_path_yaw, allYaw)
            np.save(out_path_pitch, allPitch)
            np.save(out_path_roll, allRoll)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

with h5py.File(source_main_path, 'r') as f:
    # Get frames from the h5 file
    x_test = f['x_test']
    x_train = f['x_train']
    # Get the ground truth labels
    y_test = f['y_test']
    y_train = f['y_train']

    # save_landmarks(x_test, "x_test")
    save_landmarks(x_train, "x_train")

    # Save the ground truth labels
    print("Saving ground truth labels...")
    # np.save(os.path.join(out_main_path, "y_test"), y_test)
    # print("Saved ground truth labels.")
    # print(os.path.join(out_main_path, "y_test"))
    np.save(os.path.join(out_main_path, "y_train"), y_train)
    print("Saved ground truth labels.")
    print(os.path.join(out_main_path, "y_train"))
    




