import os
from glob import glob
import numpy as np
from pathlib import Path
import cv2
import h5py
from scipy import signal
import torch
from torch import optim
import torch.nn.functional as F

from models import SyncNet_color as SyncNet
from hparams import hparams
import color_syncnet_train as color_syncnet_train
from lmks_audio_eval import cropped_mel, accuracy
import landmarks_audio as audio


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

# Iterate through all of files in all of the folders in the dataset
source_main_path = "/home/ksw38/.cache/kagglehub/datasets/adrianlubitz/vvadlrs3/versions/4/faceImages_small.h5"
# source_main_path = "/home/ksw38/.cache/kagglehub/datasets/adrianlubitz/vvadlrs3/versions/4/lipImages.h5"
out_main_path = "/home/ksw38/groups/grp_landmarks/nobackup/autodelete/landmarks_vvadlrs3/main"

checkpoint_path = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints/checkpoint_step000510000.pth"
checkpoint_path = "/home/ksw38/RVL/color_syncnet/Wav2Lip/lipsync_expert.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
syncnet_T = 5
start_frame_num = 0

model = SyncNet().to(device)

print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                    lr=hparams.syncnet_lr)
color_syncnet_train.load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False) ## For Testing only, uncomment for eval
model.eval()

##############################
# Prepare mel spectrograms
##############################
## Mel will be ignored so, we choose silence as a placeholder
silence = torch.zeros(16000)  # 1 second at 16kHz
white_noise = torch.randn(16000)
batch_size = 1
silent_mel = cropped_mel(silence, start_frame_num=0).to(device) # shape: (1, Mel, Time)
silent_mel = silent_mel.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, Mel, Time]

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

#########################################
# Load the dataset and evaluate
#########################################
data_limit = None # For testing, set to None for full dataset
with h5py.File(source_main_path, 'r') as f:
    # Get frames from the h5 file
    x_test = f['x_test']
    x_train = f['x_train']
    # Get the ground truth labels
    y_test = f['y_test']
    y_train = f['y_train']

    silent_losses = []
    silent_fconfms = []
    ys = []

    for i, frames in enumerate(x_test):
        frames = frames[start_frame_num:start_frame_num+syncnet_T] ## Full Video
        # still face frames, first frame copied 5 times
        still_face_frames = frames[0:1].repeat(syncnet_T, axis=0)
        
        y = torch.FloatTensor([y_test[i]]).to(device).unsqueeze(0)  # Convert to tensor and add batch dimension
        ys.append(y_test[i])

        x = np.concatenate(frames, axis=2)/255
        x = x.transpose(2, 0, 1)
        x = x[:, x.shape[1]//2:]
        x = torch.FloatTensor(x)
        x = x.unsqueeze(0)  # Add batch dimension
        x = x.to(device)
        silent_a, silent_v = model(silent_mel, x)

        still_x = np.concatenate(still_face_frames, axis=2)/255
        still_x = still_x.transpose(2, 0, 1)
        still_x = still_x[:, still_x.shape[1]//2:]
        still_x = torch.FloatTensor(still_x)
        still_x = still_x.unsqueeze(0)  # Add batch dimension
        still_x = still_x.to(device)  
        still_a, still_v = model(silent_mel, still_x)

        silent_loss = F.cosine_similarity(silent_v, still_v)
        silent_losses.append(silent_loss.item())
        silent_fconfm = computeDist(silent_v, still_v, vshift=15)
        silent_fconfms.append(silent_fconfm.item())

        if data_limit is not None and i >= data_limit:
            break
    
    print("Test threshold on test set")
    print("Silent losses:")
    best_accuracy(silent_losses, ys)
    print("\tFlip=True")
    best_accuracy(silent_losses, ys, flip=True)

    print("Silent fconfms:")
    best_accuracy(silent_fconfms, ys)
    print("\tFlip=True")
    best_accuracy(silent_fconfms, ys, flip=True)

    silent_losses_train = []
    silent_fconfms_train = []
    ys_train = []
    for i, frames in enumerate(x_train):
        frames = frames[start_frame_num:start_frame_num+syncnet_T] ## Full Video
        # still face frames, first frame copied 5 times
        still_face_frames = frames[0:1].repeat(syncnet_T, axis=0)
        
        y = torch.FloatTensor([y_train[i]]).to(device).unsqueeze(0)  # Convert to tensor and add batch dimension
        ys_train.append(y_train[i])

        x = np.concatenate(frames, axis=2)/255
        x = x.transpose(2, 0, 1)
        x = x[:, x.shape[1]//2:]
        x = torch.FloatTensor(x)
        x = x.unsqueeze(0)  # Add batch dimension
        x = x.to(device)
        silent_a, silent_v = model(silent_mel, x)

        still_x = np.concatenate(still_face_frames, axis=2)/255
        still_x = still_x.transpose(2, 0, 1)
        still_x = still_x[:, still_x.shape[1]//2:]
        still_x = torch.FloatTensor(still_x)
        still_x = still_x.unsqueeze(0)  # Add batch dimension
        still_x = still_x.to(device)  
        still_a, still_v = model(silent_mel, still_x)

        silent_loss = F.cosine_similarity(silent_v, still_v)
        silent_losses_train.append(silent_loss.item())
        silent_fconfm = computeDist(silent_v, still_v, vshift=15)
        silent_fconfms_train.append(silent_fconfm.item())

        if data_limit is not None and i >= data_limit:
            break

    print()
    print("Training threshold on train set")
    print("Silent losses:")
    _, sil_thresh = best_accuracy(silent_losses_train, ys_train)
    print("\tFlip=True")
    _, sil_thresh_flip = best_accuracy(silent_losses_train, ys_train, flip=True)

    print("Silent fconfms:")
    _, sil_fconfm_thresh = best_accuracy(silent_fconfms_train, ys_train)
    print("\tFlip=True")
    _, sil_fconfm_thresh_flip = best_accuracy(silent_fconfms_train, ys_train, flip=True)

    print(f"Silent loss threshold: {sil_thresh}, flip: {sil_thresh_flip}")
    print(f"Silent fconfm threshold: {sil_fconfm_thresh}, flip: {sil_fconfm_thresh_flip}")

    print("Test Train Thresholds")
    print("Silent losses:")
    sil_results = [0.0 if loss < sil_thresh else 1.0 for loss in silent_losses_train]
    sil_acc = accuracy(ys_train, sil_results)
    print(f"Silent losses accuracy: {sil_acc}")
    sil_results = [1.0 if loss < sil_thresh_flip else 0.0 for loss in silent_losses_train]
    sil_acc_flip = accuracy(ys_train, sil_results)
    print(f"Silent losses accuracy (flip): {sil_acc_flip}")
    print("Silent fconfms:")
    sil_fconfm_results = [0.0 if loss < sil_fconfm_thresh else 1.0 for loss in silent_fconfms_train]
    sil_fconfm_acc = accuracy(ys_train, sil_fconfm_results)
    print(f"Silent fconfms accuracy: {sil_fconfm_acc}")
    sil_fconfm_results = [1.0 if loss < sil_fconfm_thresh_flip else 0.0 for loss in silent_fconfms_train]
    sil_fconfm_acc_flip = accuracy(ys_train, sil_fconfm_results)
    print(f"Silent fconfms accuracy (flip): {sil_fconfm_acc_flip}")

    print()
    print("Train threshold on test set")
    print("Silent losses:")
    sil_results = [0.0 if loss < sil_thresh else 1.0 for loss in silent_losses]
    sil_acc = accuracy(ys, sil_results)
    print(f"Silent losses accuracy: {sil_acc}")
    sil_results = [1.0 if loss < sil_thresh_flip else 0.0 for loss in silent_losses]
    sil_acc_flip = accuracy(ys, sil_results)
    print(f"Silent losses accuracy (flip): {sil_acc_flip}")
    print("Silent fconfms:")
    sil_fconfm_results = [0.0 if loss < sil_fconfm_thresh else 1.0 for loss in silent_fconfms]
    sil_fconfm_acc = accuracy(ys, sil_fconfm_results)
    print(f"Silent fconfms accuracy: {sil_fconfm_acc}")
    sil_fconfm_results = [1.0 if loss < sil_fconfm_thresh_flip else 0.0 for loss in silent_fconfms]
    sil_fconfm_acc_flip = accuracy(ys, sil_fconfm_results)
    print(f"Silent fconfms accuracy (flip): {sil_fconfm_acc_flip}")
