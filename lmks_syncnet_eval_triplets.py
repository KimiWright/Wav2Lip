from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_landmarks_gru2 as SyncNet
import landmarks_audio as audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
import math
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

from collections import defaultdict
from os import path

import re
from models.lmks_only import lmks_only
from models.audio_only import audio_only
from models import SyncNet_landmarks_gru2 as SyncNet
import landmarks_syncnet_train_gru2 as gru2

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def eval_model(test_data_loader, device, model):
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):

            model.eval()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            losses.append(loss.item())

            if step > eval_steps: break

        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)

        return
    
if __name__ == "__main__":
    checkpoint_dir = 'landmarks_checkpoints_gru2'
    checkpoint_dir = "triplets_checkpoints"
    checkpoint_path = None

    print("Loading checkpoint path")
    if checkpoint_path  is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    device = 'cpu'# torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    test_dataset = gru2.Dataset('val')

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=8)

    eval_model(test_data_loader, device, model)