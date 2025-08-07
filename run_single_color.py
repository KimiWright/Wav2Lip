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
from lmks_audio_eval import cropped_mel, accuracy
from models import SyncNet_color as SyncNet

from pathlib import Path
import cv2
import time

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

