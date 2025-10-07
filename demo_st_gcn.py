from pathlib import Path
import cv2
import numpy as np
import os
import time, pdb, argparse, subprocess
import torch
import torch.nn.functional as F
import st_gcn_test as st
from models import build_adjacency
import make_mel_embedding as mme

# ==================== LOAD PARAMS ====================
parser = argparse.ArgumentParser(description = "SyncNet")

parser.add_argument('--tmp_dir', type=str, default="kimi/pytmp", help='')
parser.add_argument('--reference', type=str, default="demo", help='')
parser.add_argument('--checkpoint_st_gcn', type=str, default="checkpoints_st_gcn_norot")
parser.add_argument('--checkpoint_audio', type=str, default="checkpoints_audio_norot")
parser.add_argument('--comparison_embedding', type=str, default="kimi/babble_embedding.npy")
parser.add_argument('--videofile', type=str, default="kimi/kimi_test.mp4", help='')

opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

checkpoint_st_gcn = opt.checkpoint_st_gcn
checkpoint_audio = opt.checkpoint_audio
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
from kimi.facetools import genMediapipeInfo, clearMediapipeInfo
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

_, lmks, allYaw, allPitch, allRoll = genMediapipeInfo(frames) #this does the landmark extraction and a bunch of normalization
clearMediapipeInfo()

lmks = np.swapaxes(lmks, 1,2)
lmks = norm(lmks)

x_roll = np.array(allRoll)
x_pitch = np.array(allPitch)
x_yaw = np.array(allYaw)
x_rot = torch.FloatTensor(np.vstack((x_roll, x_pitch, x_yaw)))
x_lmks = np.swapaxes(lmks, 1,2)
x = torch.FloatTensor(x_lmks)

# =================== LOAD MODEL ============================
first_lmks = x[0].T
edges = st.knn_edges(first_lmks)
num_lmks = first_lmks.shape[0]
A = build_adjacency(num_lmks, edges)
V = num_lmks

st_gcn_model_norot, audio_model_norot = st.load_stgcn_and_audio_models(checkpoint_st_gcn, checkpoint_audio, A, V, use_cuda=use_cuda, rotation=False)

# ================= MAKE COMP EMB ===========================

comp_noise = "independent_scripts/babble_noise.wav"
comp_mel = mme.generate_mel_from_path(comp_noise).unsqueeze(0).to(device)
# print(comp_mel.shape)
# babble_mel = mme.generate_babble_mel()
# print(babble_mel.shape)
audio_model_norot = audio_model_norot.eval().to(device)
comp_emb = audio_model_norot(comp_mel)

# ============= EVALUATE MODEL AND WRITE VIDEO ===============
vidName_root = "video_sync_lmks_norm"
vidNameTrimmed = vidName_root + "_trimmed.mp4"
vidName = vidName_root + ".mp4"
vidWriter = cv2.VideoWriter(os.path.join(opt.tmp_dir,opt.reference,vidName), cv2.VideoWriter_fourcc(*'mp4v'), 25, (frames[0].shape[1],frames[0].shape[0]))
vidWriterTrimmed = cv2.VideoWriter(os.path.join(opt.tmp_dir,opt.reference,vidNameTrimmed), cv2.VideoWriter_fourcc(*'mp4v'), 25, (frames[0].shape[1],frames[0].shape[0]))
threshold = 0.580
chunk_len = 5
mid_chunk = round(chunk_len/2)
print(os.path.join(opt.tmp_dir,opt.reference,vidName))


st_gcn_model_norot.eval().to(device)
vals = []
frame_idx_vals = []
with torch.no_grad():
    for i in range(num_frames-chunk_len+1):
        image = frames[i+mid_chunk]
        input = x[i:i+chunk_len].unsqueeze(0).permute(0, 2, 1, 3).to(device).to(torch.float32)
        face_emb = st_gcn_model_norot(input).to(device).mean(dim=1)
        sim = F.cosine_similarity(comp_emb.to(device), face_emb)
        vals.append(sim.item())
        frame_idx_vals.append(i+mid_chunk)

        cv2.putText(image, '%.2f'%(sim-threshold), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
plt.savefig("kimi/demo_norm.png")