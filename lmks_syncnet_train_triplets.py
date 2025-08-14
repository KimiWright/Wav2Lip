from os.path import dirname, join, basename, isfile
from tqdm import tqdm

# from models import SyncNet_landmarks_gru2 as SyncNet
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
# from models.lmks_only import lmks_only
# from models.audio_only import audio_only
# from models import SyncNet_landmarks_gru2 as SyncNet
from models.lmks_only_attn import lmks_only_attn as lmks_only
from models.audio_only_attn import audio_only_attn as audio_only
from models import SyncNet_landmarks_attn as SyncNet


parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed landmarks for LRS2 dataset", default='/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main/')
parser.add_argument('--video_root', help='Root folder of the videos of the LRS2 dataset', default='/home/ksw38/groups/grp_lip/nobackup/autodelete/datasets/fslgroup/grp_lip/compute/datasets/LRS2/preprocessedRetinaface/lrs2/lrs2_video_seg24s/mvlrs_v1/main/')
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default='attn_checkpoints', type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)

args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = False#torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16
ID_LEN = 5 #The number of digits in the id in the file name
# The stradegy in the color_syncnet_train.py is to use the name of the mp4 files as the id and find the correspoinding mel and image files

def get_npy_list(data_root, split):
    filelist = []

    with open('landmarks_filelists/{}.txt'.format(split)) as f:
        for line in f:
            line = line.strip()
            if ' ' in line: line = line.split()[0]
            filelist.append(os.path.join(data_root, line))     

    grouped_files = defaultdict(list)
    for file in filelist:
        basename = path.basename(file)
        folder = path.basename(path.dirname(file))
        group_key = f"{folder}_{basename[:5]}"  # First 5 characters
        grouped_files[group_key].append(file)
    groups = list(grouped_files.values())
    return groups

class Triplet_Dataset(object):
    def __init__(self, split):
        # self.all_videos = get_npy_list(args.data_root, split)
        self.all_videos = get_image_list(args.video_root, split)

    def get_frame_id(self, frame):
        # return int(basename(frame).split('.')[0][0:ID_LEN])
        frame_name = basename(frame).split('.')[0]
        frame_digits = re.sub(r'\D', '', frame_name)
        return int(frame_digits)

    def get_window_npy(self, data, start_id=0):
        if start_id + syncnet_T < len(data):
            return data[start_id : start_id + syncnet_T]
        else:
            return None

    def crop_audio_window(self, spec, start_frame_num):
        
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        # Syncnet is set up randomly sync or not sync a video, that is part of why they take out 5 frame chunks
        while 1:
            # choose a random video
            idx = random.randint(0, len(self.all_videos) - 1)

            # find the path to the video at index idx
            vidname = self.all_videos[idx]
            # keep the path and filename of the video, but remove the extension (for finding the .wav file)
            vidname_no_ext = os.path.splitext(vidname)[0]

            # 5 digit id
            vidname_file = os.path.splitext(os.path.basename(vidname))[0]
            # video and landmarks folder name (log numberical id)
            vidname_folder = os.path.basename(os.path.dirname(vidname))
            # landmarks file with the 5 digit id, but not the lmks, roll, pitch, yaw endings
            npy_head = join(args.data_root, vidname_folder, vidname_file)

            # get all of the npy files corresponding to the video
            npy_files = []
            endings = ['_lmks.npy', '_roll.npy', '_pitch.npy', '_yaw.npy']
            for ending in endings:
                npy_file = npy_head + ending
                if not isfile(npy_file):
                    continue
                npy_files.append(npy_file)

            # retrive the data from the npy files
            npy_data = []
            for npy_file in npy_files:
                try:
                    npy_data.append(np.load(npy_file))
                except Exception as e:
                    print(f"Error loading npy file {npy_file}: {e}")
                    break
            if len(npy_data) != 4:
                continue

            num_frames = npy_data[0].shape[0]

            if num_frames <= 3 * syncnet_T:
                continue
            
            # get two random integers from 0 to num_frames - syncnet_T for the start of the true and false windows
            start_id = random.randint(0, num_frames - syncnet_T)
            wrong_start_id = random.randint(0, num_frames - syncnet_T)
            while wrong_start_id == start_id:
                wrong_start_id = random.randint(0, num_frames - syncnet_T)

            window_fnames = []
            window_fnames_wrong = []
            for npy_datum in npy_data:
                # get the window of npy data from start_id to start_id + syncnet_T
                window_npy = self.get_window_npy(npy_datum, start_id=start_id)
                window_npy_wrong = self.get_window_npy(npy_datum, start_id=wrong_start_id)
                if window_npy_wrong is None:
                    break
                if window_npy is None:
                    break
                window_fnames.append(window_npy)
                window_fnames_wrong.append(window_npy_wrong)

            # If the npy data is not the right length, skip this video
            if len(window_fnames) != 4:
                continue
            if len(window_fnames_wrong) != 4:
                continue

            # Get the mel spectrogram from the wav file
            try:
                wavpath = vidname_no_ext + ".wav"
                if not isfile(wavpath):
                    continue           
                wav = audio.load_wav(wavpath, hparams.sample_rate)
                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), start_id)

            if (mel.shape[0] != syncnet_mel_step_size):
                continue
            
            # Reshape and concatenate the npy data
            x_lmks = window_fnames[0].reshape(syncnet_T, -1)
            x_roll = window_fnames[1][:, None]
            x_pitch = window_fnames[2][:, None]
            x_yaw = window_fnames[3][:, None]
            x = np.concatenate([x_lmks, x_roll, x_pitch, x_yaw], axis=1)

            x_wrong_lmks = window_fnames_wrong[0].reshape(syncnet_T, -1)
            x_wrong_roll = window_fnames_wrong[1][:, None]
            x_wrong_pitch = window_fnames_wrong[2][:, None]
            x_wrong_yaw = window_fnames_wrong[3][:, None]
            x_wrong = np.concatenate([x_wrong_lmks, x_wrong_roll, x_wrong_pitch, x_wrong_yaw], axis=1)

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return mel, x, x_wrong


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

def combine_models_and_save_checkpoint(face_model, audio_model, optimizer, step, checkpoint_dir, epoch):
    combined_state_dict = {}
    face_state_dict = face_model.state_dict()
    audio_state_dict = audio_model.state_dict()

    for k, v in face_state_dict.items():
        combined_state_dict['face.' + k] = v
    for k, v in audio_state_dict.items():
        combined_state_dict['audio.' + k] = v

    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": combined_state_dict,
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved combined checkpoint:", checkpoint_path)

def triplet_loss(anchor, positive, negative, margin=0.2):
    pos_sim = F.cosine_similarity(anchor, positive)
    neg_sim = F.cosine_similarity(anchor, negative)
    loss = F.relu(pos_sim - neg_sim + margin)
    return loss.mean()

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

def train(device, face_model, audio_model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None, scheduler=None):
    global global_step, global_epoch
    resumed_step = global_step

    while global_epoch < nepochs:
        running_loss = 0.0
        prog_bar = tqdm(enumerate(train_data_loader))
        
        for step, (mel, x, x_wrong) in enumerate(tqdm(test_data_loader)):
            mel = mel.to(torch.float32).to(device)
            x = x.to(torch.float32).to(device)
            x_wrong = x_wrong.to(torch.float32).to(device)

            audio_model.train()
            face_model.train()

            anchor_emb = audio_model(mel)
            pos_emb = face_model(x)
            neg_emb = face_model(x_wrong)

            loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()

            # Step the scheduler if it is provided
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                combine_models_and_save_checkpoint(face_model, audio_model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, device, face_model, audio_model)

            prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)))
        print(f"Global_epoch: {global_epoch}")
        global_epoch += 1

def eval_model(test_data_loader, device, face_model, audio_model):
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (mel, x, x_wrong) in enumerate(tqdm(test_data_loader)):
            audio_model.eval()
            face_model.eval()
            mel = mel.to(torch.float32).to(device)
            x = x.to(torch.float32).to(device)
            x_wrong = x_wrong.to(torch.float32).to(device)

            anchor_emb = audio_model(mel)
            pos_emb = face_model(x)
            neg_emb = face_model(x_wrong)

            loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
            losses.append(loss.item())

            if step >= eval_steps:
                break
        avg_loss = np.mean(losses)
        print(avg_loss)
        return avg_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = Triplet_Dataset('val')
    train_dataset = Triplet_Dataset('train')
    checkpoint_path = args.checkpoint_path
    checkpoint_dir = args.checkpoint_dir

    if checkpoint_path is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
    print("Loading checkpoint from:", checkpoint_path)

    batch_size = hparams.syncnet_batch_size
    num_workers = 1 #8
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=num_workers, drop_last=True)
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers, drop_last=True)

    

    face_model = load_partial_model(checkpoint_path, device=device, startswith='face')
    audio_model = load_partial_model(checkpoint_path, device=device, startswith='audio')

    optimizer = optim.Adam(list(audio_model.parameters()) + list(face_model.parameters()),
                    lr=hparams.syncnet_lr, weight_decay=1e-5)

    train(device, face_model, audio_model, train_data_loader, test_data_loader, optimizer=optimizer,
          checkpoint_dir=checkpoint_dir, checkpoint_interval=hparams.syncnet_checkpoint_interval, nepochs=hparams.nepochs)

    