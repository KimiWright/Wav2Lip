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
# from lmks_audio_eval import cropped_mel, accuracy
# import landmarks_audio as audio
from models import SyncNet_color as SyncNet

from pathlib import Path
import cv2
import time

start_time = 0
frame_limit = 5

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

checkpoint_dir = 'checkpoints'
checkpoint_path = None

########################
# Mel
########################
# Switch to importing kimi/silent_mel_color.npy
silent_mel = np.load('kimi/silent_mel_color.npy')
silent_mel = torch.FloatTensor(silent_mel).to(device)

#########################
# Model Functions
#########################

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model

def process_video(video_path):
    global start_time
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video FPS: {fps}")
    frames = []
    while True:
        print("Hello")
        ret, frame = cap.read()
        print(f"Processed frame: {len(frames)}")
        if not ret:
            break
        frames.append(frame)
        
    cap.release()
    if frame_limit is not None and len(frames) > frame_limit:
        frames = frames[:frame_limit]

    cap.release()
    x = np.concatenate(frames, axis=2)/255
    x = x.transpose(2, 0, 1)
    x = x[:, x.shape[1]//2:]
    x = torch.FloatTensor(x)
    x = x.unsqueeze(0).to(device)  # Add batch dimension and move to device
    return x

if __name__ == "__main__":
    start_time = time.time()
    print(device)

    print("Loading checkpoint path")
    if checkpoint_path  is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
    print(f"Using checkpoint: {checkpoint_path}")

    model = SyncNet().to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                        lr=hparams.syncnet_lr)
    load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)
    model.eval()

    video_path = "kimi/00001.mp4" # 35 frames
    # video_path = "/home/dj/RVL_syncnet/data/kimi_test2.mp4"
    video_path = "kimi/video_sync.mp4"

    x = process_video(video_path)

    print(f"Processed video shape: {x.shape}")
    silent_a, silent_v = model(silent_mel, x)

