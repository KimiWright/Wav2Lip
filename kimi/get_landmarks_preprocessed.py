import os
from glob import glob
import numpy as np
from pathlib import Path
import cv2
from facetools import genMediapipeInfo,norm_lmks

source_main_path = "/home/ksw38/RVL/color_syncnet/Wav2Lip/lrs2_preprocessed"
out_main_path = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_preprocessed/main"

folders = [f for f in os.listdir(source_main_path) if os.path.isdir(os.path.join(source_main_path, f))]

for folder in folders:
    folder_path = os.path.join(source_main_path, folder)
    for video_folder in os.listdir(folder_path):
        # Get the frames
        source_folder_path = os.path.join(source_main_path, folder, video_folder)
        img_files = glob(os.path.join(source_folder_path, "*.jpg"))

        frames = []
        for img_file in img_files:
            frame = cv2.imread(str(img_file))
            if frame is not None:
                frames.append(frame)
            else:
                print("Warning: {img_file} frame is None")

        # Generate the Landmarks
        _, lmks, allYaw, allPitch, allRoll = genMediapipeInfo(frames) #this does the landmark extraction and a bunch of normalization
        lmks = norm_lmks(lmks) # this does the final normalization

        # Save the Landmarks
        out_folder_path = os.path.join(out_main_path, folder)
        os.makedirs(out_folder_path, exist_ok=True)
        out_path_lmks = os.path.join(out_folder_path, video_folder + "_lmks.npy")
        out_path_yaw = os.path.join(out_folder_path, video_folder + "_yaw.npy")
        out_path_pitch = os.path.join(out_folder_path, video_folder + "_pitch.npy")
        out_path_roll = os.path.join(out_folder_path, video_folder + "_roll.npy")

        np.save(out_path_lmks, lmks)
        np.save(out_path_yaw, allYaw)
        np.save(out_path_pitch, allPitch)
        np.save(out_path_roll, allRoll)
