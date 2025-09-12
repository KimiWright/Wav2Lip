from pathlib import Path
import cv2
import numpy as np
import os
import time, pdb, argparse, subprocess
import torch
import torch.nn.functional as F
import run_single as run_sing

# ==================== LOAD PARAMS ====================
parser = argparse.ArgumentParser(description = "SyncNet")

parser.add_argument('--tmp_dir', type=str, default="kimi/pytmp", help='')
parser.add_argument('--reference', type=str, default="demo", help='')
parser.add_argument('--checkpoint_dir', type=str, default="triplets_checkpoints")
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--comparison_embedding', type=str, default="kimi/babble_embedding.npy")
parser.add_argument('--videofile', type=str, default="kimi/00001.mp4", help='')

opt = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_dir = opt.checkpoint_dir
checkpoint_path = opt.checkpoint_path
comp_emb_path = opt.comparison_embedding
comp_emb = torch.Tensor(np.load(comp_emb_path))
videofile = opt.videofile

# ==================== READ VIDEO =======================
# videoPath = Path("/fslgroup/grp_lip/datasets/lrs2/mvlrs_v1/main/5551009007333662603/00001.mp4")
videoPath = Path(videofile)
cap = cv2.VideoCapture(str(videoPath))
frames = []
num_frames = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    num_frames += 1
cap.release()

# ================== GENERATE LANDMARKS ===================
from kimi.facetools import genMediapipeInfo,norm_lmks
_, lmks, allYaw, allPitch, allRoll = genMediapipeInfo(frames) #this does the landmark extraction and a bunch of normalization
lmks = norm_lmks(lmks) # this does the final normalization
#save the lmks and the yaw, pitch, roll to a filefrom facetools import genMediapipeInfo,norm_lmks
x_lmks = lmks.reshape(num_frames, -1)
x_roll = np.array(allRoll)[:, None]
x_pitch = np.array(allPitch)[:, None]
x_yaw = np.array(allYaw)[:, None]
x_video = torch.Tensor(np.concatenate([x_lmks, x_roll, x_pitch, x_yaw], axis=1))

# =================== LOAD MODEL ============================
print("Loading checkpoint path")
if checkpoint_path  is None:
    checkpoint_path = os.listdir(checkpoint_dir)[-1]
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

lmks_model = run_sing.load_face_model(checkpoint_path, device=device)
lmks_model.eval()

# ============= EVALUATE MODEL AND WRITE VIDEO ===============
# practice_array = []
# for i in range(num_frames):
#     practice_array.append(i)
# print(practice_array)
vidWriter = cv2.VideoWriter(os.path.join(opt.tmp_dir,opt.reference,'video_sync_lmks.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (frames[0].shape[1],frames[0].shape[0]))
vidWriterTrimmed = cv2.VideoWriter(os.path.join(opt.tmp_dir,opt.reference,'video_sync_trimmed_lmks.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (frames[0].shape[1],frames[0].shape[0]))
threshold = 0
chunk_len = 1
mid_chunk = round(chunk_len/2)
print(os.path.join(opt.tmp_dir,opt.reference,'video_sync_lmks.mp4'))
# x_video = torch.Tensor(x_video).unsqueeze(0).to(device).to(torch.float32)
lmks_model = lmks_model.to(device)

vals = []
frame_idx_vals = []
with torch.no_grad():
    for i in range(num_frames-chunk_len+1):
        # conf = fconfm[i-4]
        image = frames[i+mid_chunk]
        # print(practice_array[i:i+chunk_len])
        # print(practice_array[i+mid_chunk])
        input = x_video[i:i+chunk_len].unsqueeze(0).to(device).to(torch.float32)
        face_emb = lmks_model(input).to(device)
        sim = F.cosine_similarity(comp_emb.to(device), face_emb)
        vals.append(sim.item())
        frame_idx_vals.append(i+mid_chunk)

        cv2.putText(image, '%.2f'%sim, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if sim>threshold:
            vidWriterTrimmed.write(image)
        vidWriter.write(image)
vidWriter.release()
vidWriterTrimmed.release()

import matplotlib.pyplot as plt
plt.plot(frame_idx_vals, vals, marker='o')
title="Cosine Similarity values for Lmks Syncnet"
plt.title(title)
plt.xlabel("Frame Number")
plt.ylabel("Cosine Similarity")
plt.grid(True)
plt.savefig("kimi/demo.png")