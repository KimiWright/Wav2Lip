import os
from glob import glob
import numpy as np
from pathlib import Path
import cv2
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from facetools import genMediapipeInfo,norm_lmks, clearMediapipeInfo

def norm(lmks):
    lmks_norm = np.zeros_like(lmks, dtype=np.float32)
    for t in range(lmks.shape[0]):
        frame = lmks[t]
        min_xy = frame.min(axis=0)
        max_xy = frame.max(axis=0)
        scale = (max_xy - min_xy).max() / 2.0
        center = 0#(max_xy + min_xy) / 2.0
        if scale < 1e-6:
            scale = 1.0
        lmks_norm[t] = (frame - center) / scale
    return lmks_norm

# Iterate through all of files in all of the folders in the dataset
source_main_path = "/fslgroup/grp_lip/datasets/lrs2/mvlrs_v1/main/"
out_main_path = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_norm/main"
source_main_path = "/fslgroup/grp_lip/datasets/lrs2/mvlrs_v1/pretrain"
out_main_path = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_norm/pretrain_partial"
folders = [f for f in os.listdir(source_main_path) if os.path.isdir(os.path.join(source_main_path, f))]
# folders = folders[2424:len(folders)]

num_skipped_files = 0
skipped_files = []
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

        if all(Path(p).exists() for p in [out_path_lmks, out_path_yaw, out_path_pitch, out_path_roll]):
            continue

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
        _, lmks, allYaw, allPitch, allRoll = genMediapipeInfo(frames) #this does the landmark extraction and a bunch of normalization
        clearMediapipeInfo()
        try:
            lmks = np.swapaxes(lmks, 1,2)
        except Exception as e:
            print(f"\nError: {e}")
            print(f"on file {file}\n")
            num_skipped_files += 1
            skipped_files.append(file)
            continue

        lmks = norm(lmks)
        
        # Save the landmarks
        np.save(out_path_lmks, lmks)
        np.save(out_path_yaw, allYaw)
        np.save(out_path_pitch, allPitch)
        np.save(out_path_roll, allRoll)

print(f"\n{num_skipped_files} were skipped")
print(skipped_files)