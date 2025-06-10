import os
from glob import glob
import numpy as np
from pathlib import Path
import cv2
import h5py
import torch
from torch import optim

from models import SyncNet_color as SyncNet
from hparams import hparams
import color_syncnet_train as color_syncnet_train
from lmks_audio_eval import cropped_mel, accuracy

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
out_main_path = "/home/ksw38/groups/grp_landmarks/nobackup/autodelete/landmarks_vvadlrs3/main"

checkpoint_dir = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints"
checkpoint_path = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints/checkpoint_step000510000.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
syncnet_T = 5
start_frame_num = 0

model = SyncNet().to(device)

print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                    lr=hparams.syncnet_lr)
color_syncnet_train.load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)
model.eval()

silence = torch.zeros(16000)  # 1 second at 16kHz
white_noise = torch.randn(16000)
batch_size = 1
silent_mel = cropped_mel(silence, start_frame_num=0).to(device) # shape: (1, Mel, Time)
silent_mel = silent_mel.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, Mel, Time]
white_noise_mel = cropped_mel(white_noise, start_frame_num=0).to(device) # shape: (1, Mel, Time)
white_noise_mel = white_noise_mel.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, Mel, Time]


silent_losses = []
white_noise_losses = []
ys = []
with h5py.File(source_main_path, 'r') as f:
    # Get frames from the h5 file
    x_test = f['x_test']
    x_train = f['x_train']
    # Get the ground truth labels
    y_test = f['y_test']
    y_train = f['y_train']

    
    for i, frames in enumerate(x_test):
        frames = frames[start_frame_num:start_frame_num+syncnet_T]
        y = torch.FloatTensor([y_test[i]]).to(device).unsqueeze(0)  # Convert to tensor and add batch dimension
        ys.append(y_test[i])

        x = np.concatenate(frames, axis=2)/255
        x = x.transpose(2, 0, 1)
        x = x[:, x.shape[1]//2:]
        x = torch.FloatTensor(x)
        x = x.unsqueeze(0)  # Add batch dimension
        x = x.to(device)

        silent_a, silent_v = model(silent_mel, x)
        silent_loss = color_syncnet_train.cosine_loss(silent_a, silent_v, y)
        silent_losses.append(silent_loss.item())

        white_noise_a, white_noise_v = model(white_noise_mel, x)
        white_noise_loss = color_syncnet_train.cosine_loss(white_noise_a, white_noise_v, y)
        white_noise_losses.append(white_noise_loss.item())

print("Silent losses:")
best_accuracy(silent_losses, ys)
best_accuracy(silent_losses, ys, flip=True)
print("White noise losses:")
best_accuracy(white_noise_losses, ys)
best_accuracy(white_noise_losses, ys, flip=True)