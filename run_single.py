import os
import re
import numpy as np
from tqdm import tqdm

import torch
from torch import optim
import torch.nn as nn
import torch.utils.data as data_utils

from hparams import hparams
from models.lmks_only import lmks_only
from models.audio_only import audio_only

from facetools import genMediapipeInfo, norm_lmks

from pathlib import Path
import cv2


data_root = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_vvadlrs3/main/x_test/'
ground_truth = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_vvadlrs3/main/y_test.npy'
train_data_root = '/home/ksw38/groups/grp_landmarks/nobackup/autodelete/landmarks_vvadlrs3/main/x_train/'
ground_truth_train = '/home/ksw38/groups/grp_landmarks/nobackup/autodelete/landmarks_vvadlrs3/main/y_train.npy'

checkpoint_dir = 'landmarks_checkpoints_gru2'
checkpoint_dir = "triplets_checkpoints"
checkpoint_path = None

babble_embedding_path = "kimi/babble_embedding.npy"
babble_emb = torch.Tensor(np.load(babble_embedding_path))

######################
# Model Functions
######################

def load_face_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    full_state_dict = checkpoint['state_dict']
    startswith='face'
    partial_state_dict = {k: v for k, v in full_state_dict.items() if k.startswith(startswith)}

    model = lmks_only().to(device)
    
    missing, unexpected = model.load_state_dict(partial_state_dict, strict=False)
    if missing:
        print("Missing keys in the state_dict:", missing)
    if unexpected:
        print("Unexpected keys in the state_dict:", unexpected)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    return model

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    d = (d + 1) / 2 # Normalize to [0, 1]
    loss = logloss(d.unsqueeze(1), y)

    return loss

def process_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    _, lmks, allYaw, allPitch, allRoll = genMediapipeInfo(frames)
    lmks = norm_lmks(lmks) # this does the final normalization
    return len(frames), lmks, np.array(allYaw), np.array(allPitch), np.array(allRoll)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shuffle_dataset = False
    num_workers = 1
    threshold = .72

    print("Loading checkpoint path")
    if checkpoint_path  is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    lmks_model = load_face_model(checkpoint_path, device=device)
    lmks_model.eval()

    video_path = "kimi/00001.mp4" # 35 frames

    num_frames, lmks, allYaw, allPitch, allRoll = process_video(video_path)
    print(f"Processing video: {video_path} with {num_frames} frames")
    x_lmks = lmks.reshape(num_frames, -1)
    x_roll = allRoll[:, None]
    x_pitch = allPitch[:, None]
    x_yaw = allYaw[:, None]
    x_video = np.concatenate([x_lmks, x_roll, x_pitch, x_yaw], axis=1)

    x_video = torch.Tensor(x_video).unsqueeze(0).to(device).to(torch.float32)
    lmks_model = lmks_model.to(device)
    with torch.no_grad():
        face_emb = lmks_model(x_video).to(device)
        loss = cosine_loss(babble_emb.to(device), face_emb, torch.ones((1, 1)).to(device)) ###FIXME redo thresholds on run_statistics.py with this number instead of y
        print(f"Cosine loss: {loss.item()}")
        result = 1.0 if loss < threshold else 0.0
        print(f"Result: {result} (threshold: {threshold})")