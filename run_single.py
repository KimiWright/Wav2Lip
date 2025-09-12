import os
import re
import numpy as np
from tqdm import tqdm

import torch
from torch import optim
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F

from hparams import hparams
from models.lmks_only import lmks_only
from models.audio_only import audio_only

from facetools import genMediapipeInfo, norm_lmks

from pathlib import Path
import cv2
import time

checkpoint_dir = 'landmarks_checkpoints_gru2'
checkpoint_dir = "triplets_checkpoints"
checkpoint_path = None

babble_embedding_path = "kimi/babble_embedding.npy"
babble_emb = torch.Tensor(np.load(babble_embedding_path))

frame_limit = 5
start_time = 0

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

def process_video(video_path):
    global start_time
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # output_video = 'kimi/five_frames.mp4'
    # width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    # for i in range(5):
    #     ret, frame = cap.read()
    #     if not ret:
    #         print(f"Stopped early at frame {i}.")
    #         break
    #     out.write(frame)
    print(f"Video FPS: {fps}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if frame_limit is not None and len(frames) > frame_limit:
        frames = frames[:frame_limit]

    # start_time = time.time()
    _, lmks, allYaw, allPitch, allRoll = genMediapipeInfo(frames)
    lmks = norm_lmks(lmks) # this does the final normalization
    # print(f"Normalization took {time.time() - start_time:.2f} seconds")
    return len(frames), lmks, np.array(allYaw), np.array(allPitch), np.array(allRoll)

if __name__ == "__main__":
    print() ## Add visual seperation between starting warnings and printout
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    shuffle_dataset = False
    num_workers = 1
    threshold = 0.1

    print("Loading checkpoint path")
    if checkpoint_path  is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    lmks_model = load_face_model(checkpoint_path, device=device)
    lmks_model.eval()

    video_path = "kimi/00001.mp4" # 35 frames
    # video_path = "/home/dj/RVL_syncnet/data/kimi_test2.mp4"
    # video_path = "kimi/five_frames.mp4" # 5 frames
    video_path = "kimi/video_sync.mp4"

    num_frames, lmks, allYaw, allPitch, allRoll = process_video(video_path)
    # print(f"Processing video: {video_path} with {num_frames} frames")
    # print(f"Left function call {time.time() - start_time:.2f} seconds")
    x_lmks = lmks.reshape(num_frames, -1)
    x_roll = allRoll[:, None]
    x_pitch = allPitch[:, None]
    x_yaw = allYaw[:, None]
    x_video = np.concatenate([x_lmks, x_roll, x_pitch, x_yaw], axis=1)
    # print(f"Finished reshaping video data in {time.time() - start_time:.2f} seconds")

    x_video = torch.Tensor(x_video).unsqueeze(0).to(device).to(torch.float32)
    lmks_model = lmks_model.to(device)
    with torch.no_grad():
        model_start_time = time.time()
        face_emb = lmks_model(x_video).to(device)
        # print(f"Face embedding computed in {time.time() - model_start_time:.2f} seconds")
        loss = F.cosine_similarity(babble_emb.to(device), face_emb)
        print(f"Model run time: {time.time() - model_start_time:.2f} seconds")
        print(f"Found loss in {time.time() - start_time:.2f} seconds")
        print(f"Cosine loss: {loss.item()}")
        result = 1.0 if loss < threshold else 0.0
        print(f"Result: {result} (threshold: {threshold})")