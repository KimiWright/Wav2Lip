# from lmks_syncnet_train_triplets import *
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

data_root = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_vvadlrs3/main/x_test/'
ground_truth = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_vvadlrs3/main/y_test.npy'
train_data_root = '/home/ksw38/groups/grp_landmarks/nobackup/autodelete/landmarks_vvadlrs3/main/x_train/'
ground_truth_train = '/home/ksw38/groups/grp_landmarks/nobackup/autodelete/landmarks_vvadlrs3/main/y_train.npy'

global_step_finetune = 0
global_epoch_finetune = 0

################################
# Get Data and support functions
################################

def get_files_list(folder_path):
    print(folder_path)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files = sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group(0)))
    return files

def match_digits(digit, filenames):
    # Check if the digit is present in any of the filenames
    for filename in filenames:
        if str(digit) not in filename:
            return False
    return True

def get_window_npy(data, syncnet_T = 5, start_id=0):
    if start_id + syncnet_T < len(data):
        return data[start_id : start_id + syncnet_T]
    else:
        return None

def get_data(data_root, ground_truth, data_point_limit=None, start_idx=0):
    all_files = get_files_list(data_root) 
    lmks_files = [f for f in all_files if f.endswith('lmks.npy')]
    roll_files = [f for f in all_files if f.endswith('roll.npy')]
    pitch_files = [f for f in all_files if f.endswith('pitch.npy')]
    yaw_files = [f for f in all_files if f.endswith('yaw.npy')]   

    y = np.load(ground_truth)
    
    data = []
    for idx in tqdm(range(len(lmks_files))):
        if idx < start_idx:
            continue

        lmks_file = lmks_files[idx]
        roll_file = roll_files[idx]
        pitch_file = pitch_files[idx]
        yaw_file = yaw_files[idx]

        npy_file_names = [lmks_file, roll_file, pitch_file, yaw_file]

        if not match_digits(idx, npy_file_names):
            continue
        npy_files = [os.path.join(data_root, f) for f in npy_file_names]
        npy_data = []
        for npy_file in npy_files:
            try:
                npy_data.append(np.load(npy_file))
            except Exception as e:
                # print(f"Error loading npy file {npy_file}: {e}")
                break
        if len(npy_data) != 4:
            continue
        # Check if the data is empty
        if any(data.size == 0 for data in npy_data):
            # print(f"Empty data in npy files: {npy_file_names}")
            continue
        
        num_frames = npy_data[0].shape[0]
            
        x_lmks = npy_data[0].reshape(num_frames, -1)
        x_roll = npy_data[1][:, None]
        x_pitch = npy_data[2][:, None]
        x_yaw = npy_data[3][:, None]
        x_video = np.concatenate([x_lmks, x_roll, x_pitch, x_yaw], axis=1)

        min_frames = 5 # minimum number of frames for the kernel size
        if x_video.shape[0] < min_frames: 
            # print(f"Video too short for kernel size {kernel_size}: {x_video.shape[0]}")
            continue  # Skip if the video is too short for the kernel size

        data.append((x_video, y[idx]))
        if data_point_limit is not None and len(data) >= data_point_limit:
            break

    return data


class Finetune_Dataset(object):
    def __init__(self, split = 'test'):
        if split == 'test':
            self.data = get_data(data_root, ground_truth)
        elif split == 'train':
            self.data = get_data(train_data_root, ground_truth_train, data_point_limit=100, start_idx=0)
        else:
            raise ValueError("Split must be 'test' or 'train'")
        
        self.not_talking = []
        self.talking = []
        for datum in self.data:
            x_video_full, y = datum
            if x_video_full is None:
                raise ValueError("x_video_full is None")
            x_video = get_window_npy(x_video_full, start_id=0)
            if x_video is not None:
                if y == 0:
                    self.not_talking.append(x_video)
                else:
                    self.talking.append(x_video)
        
    def __len__(self):
        return len(self.not_talking)
    def __getitem__(self, idx):
        postive = self.not_talking[idx]
        negative = random.choice(self.talking)
        return (postive, negative)

################
# Loss
################  
def generate_mel_for_frames(num_frames, silence = True, video_fps=hparams.fps, mel_fps=80, sample_rate=16000, hop_length=200):
    mel_frames = int(num_frames * mel_fps / video_fps)
    num_samples = (mel_frames - 1) * hop_length  # +1 mel frame per hop
    if silence:
        gen_audio = torch.zeros(num_samples)
    else:
        gen_audio = torch.randn(num_samples) # Generate white noise
    # Compute mel spectrogram
    mel = audio.melspectrogram(gen_audio).T  # [Time, Mel]
    mel = mel[:mel_frames]  # Clip to exact mel_frames
    mel = torch.FloatTensor(mel.T).unsqueeze(0)  # [1, 80, mel_frames]
    return mel
silent_mel = generate_mel_for_frames(5, silence=True).to(torch.float32).unsqueeze(0)
white_noise_mel = generate_mel_for_frames(5, silence=False).to(torch.float32).unsqueeze(0)

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
babble_mel = generate_babble_mel(num_frames=5, start_frame_num=0, video_fps=hparams.fps, mel_fps=80).to(torch.float32).unsqueeze(0)

def finetune_triplet_loss(anchor, positive, negative, margin=0.2):
    pos_sim = F.cosine_similarity(anchor, positive)
    neg_sim = F.cosine_similarity(anchor, negative)
    loss = F.relu(neg_sim - pos_sim + margin)
    return loss.mean()

################
# Save and Load Models
################

def load_partial_model(checkpoint_path, device, startswith='face'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    full_state_dict = checkpoint['state_dict']

    partial_state_dict = {k: v for k, v in full_state_dict.items() if k.startswith(startswith)}

    if startswith == 'face':
        model = lmks_only().to(device)
    elif startswith == 'audio':
        model = audio_only().to(device)
    else:
        raise ValueError("startswith must be 'face' or 'audio'")
    missing, unexpected = model.load_state_dict(partial_state_dict, strict=False)
    if missing:
        print("Missing keys in the state_dict:", missing)
    if unexpected:
        print("Unexpected keys in the state_dict:", unexpected)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    return model

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step_finetune))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def combine_models_and_save_checkpoint(face_model, audio_model, optimizer, step, checkpoint_dir, epoch):
    combined_state_dict = {}
    face_state_dict = face_model.state_dict()
    audio_state_dict = audio_model.state_dict()

    for k, v in face_state_dict.items():
        combined_state_dict['face.' + k] = v
    for k, v in audio_state_dict.items():
        combined_state_dict['audio.' + k] = v

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step_finetune))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": combined_state_dict,
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved combined checkpoint:", checkpoint_path)

################
# Training
################
def finetune_eval_model(test_data_loader, device, face_model, audio_model):
    global babble_mel
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (pos, neg) in enumerate(tqdm(test_data_loader)):
            pos = pos.to(device).to(torch.float32)
            neg = neg.to(device).to(torch.float32)

            audio_model.eval()
            face_model.eval()

            batch_size = pos.shape[0]
            anchor_emb = audio_model(babble_mel.repeat(batch_size, 1, 1, 1).to(device))
            pos_emb = face_model(pos)
            neg_emb = face_model(neg)

            loss = finetune_triplet_loss(anchor_emb, pos_emb, neg_emb)
            losses.append(loss.item())

            if step >= eval_steps:
                break
        avg_loss = np.mean(losses)
        print(avg_loss)
        return avg_loss
    
def finetune_train(device, face_model, audio_model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None, scheduler=None):
    global global_step_finetune, global_epoch_finetune, babble_mel
    resumed_step = global_step_finetune
    print(babble_mel.shape)
    while global_epoch_finetune < nepochs:
        running_loss = 0.0
        # prog_bar = tqdm(enumerate(train_data_loader))
        
        for step, (pos, neg) in enumerate(test_data_loader):
            pos = pos.to(device).to(torch.float32)
            neg = neg.to(device).to(torch.float32)

            audio_model.train()
            face_model.train()

            batch_size = pos.shape[0]
            anchor_emb = audio_model(babble_mel.repeat(batch_size, 1, 1, 1).to(device))
            pos_emb = face_model(pos)
            neg_emb = face_model(neg)

            loss = finetune_triplet_loss(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()

            # Step the scheduler if it is provided
            if scheduler is not None:
                scheduler.step()

            global_step_finetune += 1
            cur_session_steps = global_step_finetune - resumed_step
            running_loss += loss.item()

            if global_step_finetune == 1 or global_step_finetune % checkpoint_interval == 0:
                combine_models_and_save_checkpoint(face_model, audio_model, optimizer, global_step_finetune, checkpoint_dir, global_epoch_finetune)

            if global_step_finetune % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    finetune_eval_model(test_data_loader, device, face_model, audio_model)

            # prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)))
        # print(f"Global_epoch: {global_epoch}")
        global_epoch_finetune += 1


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = hparams.syncnet_batch_size
    num_workers = 1
    
    checkpoint_path = "/home/ksw38/RVL/color_syncnet/Wav2Lip/triplets_checkpoints/checkpoint_step002370000.pth"
    checkpoint_dir = "finetune_checkpoints_babble"

    if checkpoint_path is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
    print('checkpoint path: {}'.format(checkpoint_path))

    face_model = load_partial_model(checkpoint_path, device, startswith='face')
    audio_model = load_partial_model(checkpoint_path, device, startswith='audio')

    test_dataset = Finetune_Dataset('test')
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=num_workers)

    train_dataset = Finetune_Dataset('train')
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers, shuffle=True, drop_last=True)

    print(f"\nNumber of samples in finetune dataset: {len(train_dataset)}")
    print(f"\nNumber of samples in finetune test dataset: {len(test_dataset)}")

    optimizer = optim.Adam(list(audio_model.parameters()) + list(face_model.parameters()),
                    lr=hparams.syncnet_lr, weight_decay=1e-5)
    
    finetune_train(device, face_model, audio_model, train_data_loader, test_data_loader, optimizer=optimizer,
          checkpoint_dir=checkpoint_dir, checkpoint_interval=hparams.syncnet_checkpoint_interval, nepochs=hparams.nepochs)
