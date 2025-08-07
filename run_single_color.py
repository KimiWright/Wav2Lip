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
import landmarks_audio as audio
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
# Mel doesn't matter for timing, we can update to the most accurate mel later
syncnet_mel_step_size = 80  # 80 ms per step, as per SyncNet paper
def crop_audio_window(spec, start_frame_num):
        
    start_idx = int(80. * (start_frame_num / float(hparams.fps)))

    end_idx = start_idx + syncnet_mel_step_size

    return spec[start_idx : end_idx, :]

def cropped_mel(audio_tensor, start_frame_num=0):
    mel = audio.melspectrogram(audio_tensor).T # shape: (Time, Mel)
    cropped_mel = crop_audio_window(mel.copy(), start_frame_num)
    mel = torch.FloatTensor(cropped_mel.T).unsqueeze(0)  # [1, Mel, Time]
    return mel


silence = torch.zeros(16000)  # 1 second at 16kHz
white_noise = torch.randn(16000)
batch_size = 1
silent_mel = cropped_mel(silence, start_frame_num=0).to(device) # shape: (1, Mel, Time)
silent_mel = silent_mel.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, Mel, Time]

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
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if frame_limit is not None and len(frames) > frame_limit:
        frames = frames[:frame_limit]

    cap.release()

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

