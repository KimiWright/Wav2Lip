from pathlib import Path
import cv2
import numpy as np
import os
import time, pdb, argparse, subprocess
import torch
import torch.nn.functional as F
import st_gcn_test as st
from models import build_adjacency

# ==================== LOAD PARAMS ====================
parser = argparse.ArgumentParser(description = "SyncNet")

parser.add_argument('--tmp_dir', type=str, default="kimi/pytmp", help='')
parser.add_argument('--reference', type=str, default="demo", help='')
parser.add_argument('--checkpoint_st_gcn', type=str, default="checkpoints_st_gcn_norot")
parser.add_argument('--checkpoint_audio', type=str, default="checkpoints_audio_norot")
parser.add_argument('--comparison_embedding', type=str, default="kimi/babble_embedding.npy")
parser.add_argument('--videofile', type=str, default="kimi/00001.mp4", help='')

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

st_gcn_model_norot, audio_model_norot = st.load_stgcn_and_audio_models(checkpoint_st_gcn, checkpoint_st_gcn, A, V, use_cuda=use_cuda, rotation=False)

# ============= EVALUATE MODEL AND WRITE VIDEO ===============
vidWriter = cv2.VideoWriter(os.path.join(opt.tmp_dir,opt.reference,'video_sync_lmks.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (frames[0].shape[1],frames[0].shape[0]))
vidWriterTrimmed = cv2.VideoWriter(os.path.join(opt.tmp_dir,opt.reference,'video_sync_trimmed_lmks.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (frames[0].shape[1],frames[0].shape[0]))
threshold = 0
chunk_len = 1
mid_chunk = round(chunk_len/2)
print(os.path.join(opt.tmp_dir,opt.reference,'video_sync_lmks.mp4'))