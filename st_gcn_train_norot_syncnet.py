import st_gcn_test as st
from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import audio_only
from models import LandmarkSTGCNConformer
from models import build_adjacency

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

audio_model_checkpoint_dir = "checkpoints_audio_norot_facial"
st_gcn_checkpoint_dir = "checkpoints_st_gcn_norot_facial"
global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

logloss = nn.BCELoss()
# def cosine_loss(a, v, y):
#     d = F.cosine_similarity(a, v)
#     d_before = d
#     d = torch.clamp(d, min=0., max=1.)
#     if not torch.equal(d_before, d):
#         print(f"Clamped {d_before} to {d}")

#     loss = logloss(d.unsqueeze(1), y)

#     return loss

def cosine_loss(a, v, y):
    d = F.cosine_similarity(a, v)          # [-1, 1]
    d = (d + 1) / 2                        # [0, 1]
    d = torch.clamp(d, min=0., max=1.)
    d_before = d
    if not torch.equal(d_before, d):
        print(f"Clamped {d_before} to {d}")
    loss = logloss(d.unsqueeze(1), y)      # BCE works safely
    return loss

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def eval_model(test_data_loader, device, st_gcn_model, audio_model):
    eval_steps = 1400
    # eval_steps = None ## Modification ##
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (x, x_rot, mel, y) in enumerate(test_data_loader):

            st_gcn_model.eval()
            audio_model.eval()

            # Transform data to CUDA device

            mel = mel.to(device)

            x = x.permute(0, 2, 1, 3).to(device)

            lmk_feat = st_gcn_model(x)
            v = lmk_feat.mean(dim=1)
            a = audio_model(mel)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            losses.append(loss.item())

            if eval_steps is not None and step > eval_steps: break ## Modification ##

        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)

        return

def train(device, st_gcn_model, audio_model, train_data_loader, test_data_loader, st_gcn_optimizer, audio_optimizer,
          st_gcn_checkpoint_dir=None, audio_checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step
    
    while global_epoch < nepochs:
        running_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, x_rot, mel, y) in prog_bar:
            st_gcn_model.train()
            audio_model.train()
            st_gcn_optimizer.zero_grad()
            audio_optimizer.zero_grad()

            # Transform data to CUDA device

            mel = mel.to(device)

            x = x.permute(0, 2, 1, 3).to(device)

            lmk_feat = st_gcn_model(x)
            v = lmk_feat.mean(dim=1)
            a = audio_model(mel)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            loss.backward()
            st_gcn_optimizer.step()
            audio_optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    st_gcn_model, st_gcn_optimizer, global_step, st_gcn_checkpoint_dir, global_epoch)
                
            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    audio_model, audio_optimizer, global_step, audio_checkpoint_dir, global_epoch)

            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, device, st_gcn_model, audio_model)

            prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)))
        print(f"Global_epoch: {global_epoch}")
        global_epoch += 1

if __name__ == "__main__":
    print(f"Using Data from {st.data_root}")
    # Dataset and Dataloader setup
    test_dataset = st.Dataset('val')
    train_dataset = st.Dataset('train')

    num_workers = 1 #hparams.num_workers, 8
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=num_workers)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    audio_model = audio_only().to(device)
    audio_optimizer = optim.Adam([p for p in audio_model.parameters() if p.requires_grad],
                            lr=hparams.syncnet_lr, weight_decay=1e-5)
    ## FIXME: Loading code will go here ##
    
    first_point = test_dataset[0]
    (x, x_rot, mel, y) = first_point
    first_lmks = x[0].T
    # edges = st.knn_edges(first_lmks)
    print("Using Facial Edges")
    edges = st.facial_edges()
    num_lmks = first_lmks.shape[0]
    A = build_adjacency(num_lmks, edges)
    V = num_lmks
    C = 2 # x,y
    K = 1 # 1 partion, chosen arbitarily 
    
    st_gcn_model = LandmarkSTGCNConformer(
        num_nodes=V,
        A=A,
        d_model=128,
        post_linear_hidden=128,
        conformer_layers=4,
        conformer_heads=4,
        conformer_ff=256,
        conformer_conv_kernel=31
    ).to(device)

    st_gcn_optimizer = optim.Adam([p for p in st_gcn_model.parameters() if p.requires_grad],
                            lr=hparams.syncnet_lr, weight_decay=1e-5)
    
    ## FIXME: Loading code will go here ##

    # Training
    print("Beginning Training")
    train(device, st_gcn_model, audio_model, train_data_loader, test_data_loader, st_gcn_optimizer, audio_optimizer,
          st_gcn_checkpoint_dir, audio_model_checkpoint_dir, checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)