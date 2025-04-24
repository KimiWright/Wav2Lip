import os
from glob import glob
import numpy as np
from pathlib import Path
import cv2

# Iterate through all of files in all of the folders in the dataset
source_main_path = "/fslgroup/grp_lip/datasets/lrs2/mvlrs_v1/main/"
out_main_path = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main"
folders = [f for f in os.listdir(source_main_path) if os.path.isdir(os.path.join(source_main_path, f))]
folders = folders[2424:len(folders)]

for folder in folders:
    source_folder_path = os.path.join(source_main_path, folder)
    files = glob(os.path.join(source_folder_path, "*.mp4"))
    for file in files:
        # Make File Paths
        source_path = os.path.join(source_main_path, folder, file)
        folder_path = os.path.join(out_main_path, folder)
        os.makedirs(folder_path, exist_ok=True)

        file_name = os.path.splitext(os.path.basename(file))[0]
        out_path_lmks = os.path.join(out_main_path, folder, file_name + "_lmks")
        out_path_yaw = os.path.join(out_main_path, folder, file_name + "_yaw")
        out_path_pitch = os.path.join(out_main_path, folder, file_name + "_pitch")
        out_path_roll = os.path.join(out_main_path, folder, file_name + "_roll")

        # Get the landmarks (Shad's code)
        videoPath = file
        cap = cv2.VideoCapture(str(videoPath))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        from facetools import genMediapipeInfo,norm_lmks
        _, lmks, allYaw, allPitch, allRoll = genMediapipeInfo(frames) #this does the landmark extraction and a bunch of normalization
        lmks = norm_lmks(lmks) # this does the final normalization
        
        # Save the landmarks
        np.save(out_path_lmks, lmks)
        np.save(out_path_yaw, allYaw)
        np.save(out_path_pitch, allPitch)
        np.save(out_path_roll, allRoll)


