import os
from glob import glob
import numpy as np
from pathlib import Path
import cv2
from scipy import signal
import torch
from torch import optim
from torch.utils import data as data_utils
import torch.nn.functional as F

from models import SyncNet_color as SyncNet
from hparams import hparams
from lmks_audio_eval import cropped_mel, accuracy
import landmarks_audio as audio

from preprocessed_dataset import Dataset

checkpoint_path = "/home/ksw38/RVL/color_syncnet/Wav2Lip/lipsync_expert.pth"
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

def thresh_accuracy(losses, true_y, threshold, flip=False):
    if flip:
            results = [1.0 if loss < threshold else 0.0 for loss in losses]
    else:
        results = [0.0 if loss < threshold else 1.0 for loss in losses]
    acc = accuracy(true_y, results)
    return acc

def best_accuracy(losses, true_y, flip=False, thresholds=np.arange(0.0, 1.2, 0.1)):
    best_acc = 0
    best_threshold = 0
    for threshold in thresholds:
        if flip:
            results = [1.0 if loss < threshold else 0.0 for loss in losses]
        else:
            results = [0.0 if loss < threshold else 1.0 for loss in losses]
        acc = accuracy(true_y, results)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    print(f"Best accuracy: {best_acc} at threshold: {best_threshold}")
    return best_acc, best_threshold

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

####################################
# Calc Pdist
#####################################

def calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift*2+1
    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))
    dists = []
    for i in range(0,len(feat1)):
        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))
    return dists

def computeDist(feat1, feat2, vshift=15):
    dists = calc_pdist(feat1, feat2, vshift=vshift)
    mdist = torch.mean(torch.stack(dists, 1), 1)
    minval, minidx = torch.min(mdist, 0)

    mdist = mdist.detach().cpu()
    minidx = minidx.item()

    fdist = np.stack([dist[minidx].detach().cpu().numpy() for dist in dists])
    fconf = torch.median(mdist).item() - fdist
    if fconf.shape[0] < 9:
        kernel = fconf.shape[0] // 2 * 2 + 1  # Next odd number below size
    else:
        kernel = 9
    fconfm = signal.medfilt(fconf, kernel_size=kernel)


    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    return fconfm

def find_thresholds(vals, sep = 0.1): #sep = seperation of values
    min_threshold = min(vals)
    max_threshold = max(vals)
    threshold_range = np.arange(min_threshold - sep, max_threshold + sep, sep)
    return threshold_range

def eval_model(data_loader, step_limit = None):
    sim_vals = []
    fconfm_vals = []
    y_truth_vals = []
    with torch.no_grad():
        for step, (x, mel, y) in enumerate(data_loader):

            a, v = model(mel, x)
            sim = F.cosine_similarity(a, v)
            fconfm = computeDist(a, v)

            sim_vals.append(sim)
            fconfm_vals.append(fconfm)
            y_truth_vals.append(y.item())

            if step_limit is not None and step >= step_limit:
                break
    
    return y_truth_vals, sim_vals, fconfm_vals

def print_data_stats(y_vals, name=""):
    num_pos_y_vals = y_vals.count(1)
    num_neg_y_vals = y_vals.count(0)
    print(f"{name}: pos {num_pos_y_vals}, neg {num_neg_y_vals}")


syncnet_T = 5
start_frame_num = 0

model = SyncNet().to(device)

print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                    lr=hparams.syncnet_lr)
load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)
model.eval()

if __name__ == "__main__":
    test_dataset = Dataset('val')
    train_dataset = Dataset('train')
    step_limit = None
    print(f"Step Limit: {step_limit}")

    batch_size = hparams.syncnet_batch_size
    num_workers = 8
    batch_size = 1
    num_workers = 1
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=num_workers)
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers)
    
    y_truth_vals, sim_vals, fconfm_vals = eval_model(test_data_loader, step_limit=step_limit)
    y_truth_vals_train, sim_vals_train, fconfm_vals_train = eval_model(train_data_loader, step_limit=step_limit)

    print_data_stats(y_truth_vals, "Test")
    print_data_stats(y_truth_vals_train, "Train")


    print("Test threshold on the test set")
    print("\tCosine Similarity")
    sim_thresholds = find_thresholds(sim_vals)
    best_accuracy(sim_vals, y_truth_vals, flip=False, thresholds=sim_thresholds)
    best_accuracy(sim_vals, y_truth_vals, flip=True, thresholds=sim_thresholds)

    print("\tFconfm")
    fconfm_thresholds = find_thresholds(fconfm_vals)
    best_accuracy(fconfm_vals, y_truth_vals, flip=False, thresholds=fconfm_thresholds)
    best_accuracy(fconfm_vals, y_truth_vals, flip=True, thresholds=fconfm_thresholds)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Train threshold Cosine Similarity")
    sim_thresholds_train = find_thresholds(sim_vals_train)
    flip = False
    print(f"Flip = {flip}")
    print("\tTraining set")
    best_acc, thresh = best_accuracy(sim_vals_train, y_truth_vals_train, flip=flip, thresholds=sim_thresholds_train)
    acc = thresh_accuracy(sim_vals, y_truth_vals, thresh, flip)
    print(f"\tTest Set\n{acc}")

    flip = True
    print(f"Flip = {flip}")
    print("\tTraining set")
    best_acc, thresh = best_accuracy(sim_vals_train, y_truth_vals_train, flip=flip, thresholds=sim_thresholds_train)
    acc = thresh_accuracy(sim_vals, y_truth_vals, thresh, flip)
    print(f"\tTest Set\n{acc}")

    print("Train threshold Fconfm")
    fconfm_thresholds_train = find_thresholds(fconfm_vals_train)
    flip = False
    print(f"Flip = {flip}")
    print("\tTraining set")
    best_acc, thresh = best_accuracy(fconfm_vals_train, y_truth_vals_train, flip=flip, thresholds=fconfm_thresholds_train)
    acc = thresh_accuracy(fconfm_vals, y_truth_vals, thresh, flip)
    print(f"\tTest Set\n{acc}")

    flip = True
    print(f"Flip = {flip}")
    print("\tTraining set")
    best_acc, thresh = best_accuracy(fconfm_vals_train, y_truth_vals_train, flip=flip, thresholds=fconfm_thresholds_train)
    acc = thresh_accuracy(fconfm_vals, y_truth_vals, thresh, flip)
    print(f"\tTest Set\n{acc}")
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    