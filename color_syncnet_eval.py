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

##############################
# Prepare mel spectrograms
##############################


silence = torch.zeros(16000)  # 1 second at 16kHz
white_noise = torch.randn(16000)
batch_size = 1
silent_mel = cropped_mel(silence, start_frame_num=0).to(device) # shape: (1, Mel, Time)
silent_mel = silent_mel.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, Mel, Time]
white_noise_mel = cropped_mel(white_noise, start_frame_num=0).to(device) # shape: (1, Mel, Time)
white_noise_mel = white_noise_mel.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, Mel, Time]

def crop_audio_window(spec, num_frames=5, start_frame_num=0, video_fps=hparams.fps, mel_fps=80):
        mel_frames = int(num_frames * mel_fps / video_fps)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        end_idx = start_idx + mel_frames
        return spec[start_idx : end_idx, :]

babble_noise = '/home/ksw38/groups/grp_lip/nobackup/archive/datasets/speech-commands/_background_noise_/babble_noise.wav'
babble_wave = audio.load_wav(babble_noise, hparams.sample_rate)
babble_mel_global = audio.melspectrogram(babble_wave).T  # [Time, Mel]
def generate_babble_mel(num_frames=5, start_frame_num=0, video_fps=hparams.fps, mel_fps=80):
    babble_mel = crop_audio_window(babble_mel_global.copy(), num_frames=num_frames, start_frame_num=start_frame_num, video_fps=video_fps, mel_fps=mel_fps)  # Crop to the first mel step
    babble_mel = torch.FloatTensor(babble_mel.T).unsqueeze(0)  # [1, Mel, Time]
    return babble_mel
babble_mel = generate_babble_mel().to(device)  # [1, Mel, Time]
babble_mel = babble_mel.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, Mel, Time]

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

with h5py.File(source_main_path, 'r') as f:
    # Get frames from the h5 file
    x_test = f['x_test']
    x_train = f['x_train']
    # Get the ground truth labels
    y_test = f['y_test']
    y_train = f['y_train']

    silent_losses = []
    white_noise_losses = []
    babble_losses = []
    silent_fconfms = []
    white_noise_fconfms = []
    babble_fconfms = []
    ys = []
    for i, frames in enumerate(x_test):
        frames = frames[start_frame_num:start_frame_num+syncnet_T] ## Full Video
        y = torch.FloatTensor([y_test[i]]).to(device).unsqueeze(0)  # Convert to tensor and add batch dimension
        ys.append(y_test[i])

        x = np.concatenate(frames, axis=2)/255
        x = x.transpose(2, 0, 1)
        x = x[:, x.shape[1]//2:]
        x = torch.FloatTensor(x)
        x = x.unsqueeze(0)  # Add batch dimension
        x = x.to(device)
        silent_a, silent_v = model(silent_mel, x)
        # silent_loss = color_syncnet_train.cosine_loss(silent_a, silent_v, y)
        # silent_loss = color_syncnet_train.cosine_loss(silent_a, silent_v, torch.ones((1, 1)).to(device))
        silent_loss = F.cosine_similarity(silent_a, silent_v)
        silent_losses.append(silent_loss.item())
        silent_fconfm = computeDist(silent_a, silent_v, vshift=15)
        silent_fconfms.append(silent_fconfm.item())

        white_noise_a, white_noise_v = model(white_noise_mel, x)
        # white_noise_loss = color_syncnet_train.cosine_loss(white_noise_a, white_noise_v, y)
        # white_noise_loss = color_syncnet_train.cosine_loss(white_noise_a, white_noise_v, torch.ones((1, 1)).to(device))
        white_noise_loss = F.cosine_similarity(white_noise_a, white_noise_v)
        white_noise_losses.append(white_noise_loss.item())
        white_noise_fconfm = computeDist(white_noise_a, white_noise_v, vshift=15)
        white_noise_fconfms.append(white_noise_fconfm.item())

        babble_a, babble_v = model(babble_mel, x)
        # babble_loss = color_syncnet_train.cosine_loss(babble_a, babble_v, y)
        # babble_loss = color_syncnet_train.cosine_loss(babble_a, babble_v, torch.ones((1, 1)).to(device))
        babble_loss = F.cosine_similarity(babble_a, babble_v)
        babble_losses.append(babble_loss.item())
        babble_fconfm = computeDist(babble_a, babble_v, vshift=15)
        babble_fconfms.append(babble_fconfm.item())

    print("Test threshold on the test set")
    print("Silent losses:")
    best_accuracy(silent_losses, ys)
    best_accuracy(silent_losses, ys, flip=True)

    print("White noise losses:")
    best_accuracy(white_noise_losses, ys)
    best_accuracy(white_noise_losses, ys, flip=True)

    print("Babble losses:")
    best_accuracy(babble_losses, ys)
    best_accuracy(babble_losses, ys, flip=True)

    print("Silent fconfms:")
    best_accuracy(silent_fconfms, ys)
    best_accuracy(silent_fconfms, ys, flip=True)

    print("White noise fconfms:")
    best_accuracy(white_noise_fconfms, ys)
    best_accuracy(white_noise_fconfms, ys, flip=True)

    print("Babble fconfms:")
    best_accuracy(babble_fconfms, ys)
    best_accuracy(babble_fconfms, ys, flip=True)

    silent_losses_train = []
    white_noise_losses_train = []
    babble_losses_train = []
    silent_fconfms_train = []
    white_noise_fconfms_train = []
    babble_fconfms_train = []
    ys_train = []
    for i, frames in enumerate(x_train):
        frames = frames[start_frame_num:start_frame_num+syncnet_T] ## Full Video
        y = torch.FloatTensor([y_train[i]]).to(device).unsqueeze(0)  # Convert to tensor and add batch dimension
        ys_train.append(y_train[i])

        x = np.concatenate(frames, axis=2)/255
        x = x.transpose(2, 0, 1)
        x = x[:, x.shape[1]//2:]
        x = torch.FloatTensor(x)
        x = x.unsqueeze(0)  # Add batch dimension
        x = x.to(device)

        silent_a, silent_v = model(silent_mel, x)
        # silent_loss = color_syncnet_train.cosine_loss(silent_a, silent_v, y)
        # silent_loss = color_syncnet_train.cosine_loss(silent_a, silent_v, torch.ones((1, 1)).to(device))
        silent_loss = F.cosine_similarity(silent_a, silent_v)
        silent_losses_train.append(silent_loss.item())
        silent_fconfm = computeDist(silent_a, silent_v, vshift=15)
        silent_fconfms_train.append(silent_fconfm.item())

        white_noise_a, white_noise_v = model(white_noise_mel, x)
        # white_noise_loss = color_syncnet_train.cosine_loss(white_noise_a, white_noise_v, y)
        # white_noise_loss = color_syncnet_train.cosine_loss(white_noise_a, white_noise_v, torch.ones((1, 1)).to(device))
        white_noise_loss = F.cosine_similarity(white_noise_a, white_noise_v)
        white_noise_losses_train.append(white_noise_loss.item())
        white_noise_fconfm = computeDist(white_noise_a, white_noise_v, vshift=15)
        white_noise_fconfms_train.append(white_noise_fconfm.item())

        babble_a, babble_v = model(babble_mel, x)
        # babble_loss = color_syncnet_train.cosine_loss(babble_a, babble_v, y)
        # babble_loss = color_syncnet_train.cosine_loss(babble_a, babble_v, torch.ones((1, 1)).to(device))
        babble_loss = F.cosine_similarity(babble_a, babble_v)
        babble_losses_train.append(babble_loss.item())
        babble_fconfm = computeDist(babble_a, babble_v, vshift=15)
        babble_fconfms_train.append(babble_fconfm.item())

    print()
    print("Training thresholds on the training set")
    print("Silent losses:")
    _, sil_thresh = best_accuracy(silent_losses_train, ys_train)

    print("White noise losses:")
    _, wn_thresh = best_accuracy(white_noise_losses_train, ys_train)

    print("Babble losses:")
    _, ba_thresh = best_accuracy(babble_losses_train, ys_train)

    print()
    print("These should match the training losses, if they don't there are errors in this code")
    sil_results = [0.0 if loss < sil_thresh else 1.0 for loss in silent_losses_train]
    wn_results = [0.0 if loss < wn_thresh else 1.0 for loss in white_noise_losses_train]
    ba_results = [0.0 if loss < ba_thresh else 1.0 for loss in babble_losses_train]
    sil_acc = accuracy(ys_train, sil_results)
    wn_acc = accuracy(ys_train, wn_results)
    ba_acc = accuracy(ys_train, ba_results)
    print(f"Silent accuracy: {sil_acc}")
    print(f"White noise accuracy: {wn_acc}")
    print(f"Babble accuracy: {ba_acc}")
    print()
    ## Train threshold on the test set
    print("Training thresholds on the test set")
    sil_results = [0.0 if loss < sil_thresh else 1.0 for loss in silent_losses]
    wn_results = [0.0 if loss < wn_thresh else 1.0 for loss in white_noise_losses]
    ba_results = [0.0 if loss < ba_thresh else 1.0 for loss in babble_losses]
    sil_acc = accuracy(ys, sil_results)
    wn_acc = accuracy(ys, wn_results)
    ba_acc = accuracy(ys, ba_results)
    print(f"Silent accuracy: {sil_acc}")
    print(f"White noise accuracy: {wn_acc}")
    print(f"Babble accuracy: {ba_acc}")

    print()
    print("####################\nFlip\n####################")
    print()
    print("Training thresholds on the training set")
    print("Silent losses:")
    _, sil_thresh = best_accuracy(silent_losses_train, ys_train, flip=True)

    print("White noise losses:")
    _, wn_thresh = best_accuracy(white_noise_losses_train, ys_train, flip=True)

    print("Babble losses:")
    _, ba_thresh = best_accuracy(babble_losses_train, ys_train, flip=True)

    print()
    print("These should match the training losses, if they don't there are errors in this code")
    sil_results = [1.0 if loss < sil_thresh else 0.0 for loss in silent_losses_train]
    wn_results = [1.0 if loss < wn_thresh else 0.0 for loss in white_noise_losses_train]
    ba_results = [1.0 if loss < ba_thresh else 0.0 for loss in babble_losses_train]
    sil_acc = accuracy(ys_train, sil_results)
    wn_acc = accuracy(ys_train, wn_results)
    ba_acc = accuracy(ys_train, ba_results)
    print(f"Silent accuracy: {sil_acc}")
    print(f"White noise accuracy: {wn_acc}")
    print(f"Babble accuracy: {ba_acc}")
    print()
    ## Train threshold on the test set
    print("Training thresholds on the test set")
    sil_results = [1.0 if loss < sil_thresh else 0.0 for loss in silent_losses]
    wn_results = [1.0 if loss < wn_thresh else 0.0 for loss in white_noise_losses]
    ba_results = [1.0 if loss < ba_thresh else 0.0 for loss in babble_losses]
    sil_acc = accuracy(ys, sil_results)
    wn_acc = accuracy(ys, wn_results)
    ba_acc = accuracy(ys, ba_results)
    print(f"Silent accuracy: {sil_acc}")
    print(f"White noise accuracy: {wn_acc}")
    print(f"Babble accuracy: {ba_acc}")


    print()
    print("\tFconfms")
    print("Silent fconfms:")
    _, sil_thresh_fconfm = best_accuracy(silent_fconfms_train, ys_train)

    print("White noise fconfms:")
    _, wn_thresh_fconfm = best_accuracy(white_noise_fconfms_train, ys_train)

    print("Babble fconfms:")
    _, ba_thresh_fconfm = best_accuracy(babble_fconfms_train, ys_train)

    print()
    print("These should match the training fconfms, if they don't there are errors in this code")
    sil_results = [0.0 if fconfm < sil_thresh_fconfm else 1.0 for fconfm in silent_fconfms_train]
    wn_results = [0.0 if fconfm < wn_thresh_fconfm else 1.0 for fconfm in white_noise_fconfms_train]
    ba_results = [0.0 if fconfm < ba_thresh_fconfm else 1.0 for fconfm in babble_fconfms_train]
    sil_acc = accuracy(ys_train, sil_results)
    wn_acc = accuracy(ys_train, wn_results)
    ba_acc = accuracy(ys_train, ba_results)
    print(f"Silent accuracy: {sil_acc}")
    print(f"White noise accuracy: {wn_acc}")
    print(f"Babble accuracy: {ba_acc}")
    print()
    ## Train threshold on the test set
    print("Training thresholds on the test set")
    sil_results = [0.0 if fconfm < sil_thresh_fconfm else 1.0 for fconfm in silent_fconfms]
    wn_results = [0.0 if fconfm < wn_thresh_fconfm else 1.0 for fconfm in white_noise_fconfms]
    ba_results = [0.0 if fconfm < ba_thresh_fconfm else 1.0 for fconfm in babble_fconfms]
    sil_acc = accuracy(ys, sil_results)
    wn_acc = accuracy(ys, wn_results)
    ba_acc = accuracy(ys, ba_results)
    print(f"Silent accuracy: {sil_acc}")
    print(f"White noise accuracy: {wn_acc}")
    print(f"Babble accuracy: {ba_acc}")
    print()
    print("\tFlip")
    print("Training thresholds on the training set")
    print("Silent fconfms:")
    _, sil_thresh_fconfm = best_accuracy(silent_fconfms_train, ys_train, flip=True) 
    print("White noise fconfms:")
    _, wn_thresh_fconfm = best_accuracy(white_noise_fconfms_train, ys_train, flip=True)
    print("Babble fconfms:")
    _, ba_thresh_fconfm = best_accuracy(babble_fconfms_train, ys_train, flip=True)
    print()
    print("These should match the training fconfms, if they don't there are errors in this code")
    sil_results = [1.0 if fconfm < sil_thresh_fconfm else 0.0 for fconfm in silent_fconfms_train]
    wn_results = [1.0 if fconfm < wn_thresh_fconfm else 0.0 for fconfm in white_noise_fconfms_train]
    ba_results = [1.0 if fconfm < ba_thresh_fconfm else 0.0 for fconfm in babble_fconfms_train]
    sil_acc = accuracy(ys_train, sil_results)
    wn_acc = accuracy(ys_train, wn_results)
    ba_acc = accuracy(ys_train, ba_results)
    print(f"Silent accuracy: {sil_acc}")
    print(f"White noise accuracy: {wn_acc}")
    print(f"Babble accuracy: {ba_acc}")
    ## Train threshold on the test set
    print("Training thresholds on the test set")
    sil_results = [1.0 if fconfm < sil_thresh_fconfm else 0.0 for fconfm in silent_fconfms]
    wn_results = [1.0 if fconfm < wn_thresh_fconfm else 0.0 for fconfm in white_noise_fconfms]
    ba_results = [1.0 if fconfm < ba_thresh_fconfm else 0.0 for fconfm in babble_fconfms]
    sil_acc = accuracy(ys, sil_results)
    wn_acc = accuracy(ys, wn_results)
    ba_acc = accuracy(ys, ba_results)
    print(f"Silent accuracy: {sil_acc}")
    print(f"White noise accuracy: {wn_acc}")
    print(f"Babble accuracy: {ba_acc}")