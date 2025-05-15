from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_landmarks_gru as SyncNet
import landmarks_audio as audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

from collections import defaultdict
from os import path

import re

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed landmarks for LRS2 dataset", default='/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main/')
parser.add_argument('--video_root', help='Root folder of the videos of the LRS2 dataset', default='/home/ksw38/groups/grp_lip/nobackup/autodelete/datasets/fslgroup/grp_lip/compute/datasets/LRS2/preprocessedRetinaface/lrs2/lrs2_video_seg24s/mvlrs_v1/main/')
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default='landmarks_checkpoints_gru', type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)

args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
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

class Dataset(object):
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
                npy_data.append(np.load(npy_file))

            num_frames = npy_data[0].shape[0]

            if num_frames <= 3 * syncnet_T:
                continue
            
            # get two random integers from 0 to num_frames - syncnet_T for the start of the true and false windows
            start_id = random.randint(0, num_frames - syncnet_T)
            wrong_start_id = random.randint(0, num_frames - syncnet_T)
            while wrong_start_id == start_id:
                wrong_start_id = random.randint(0, num_frames - syncnet_T)

            # Choose whether this will be a true or false window
            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = start_id
            else:
                y = torch.zeros(1).float()
                chosen = wrong_start_id

            window_fnames = []
            for npy_datum in npy_data:
                # get the window of npy data from start_id to start_id + syncnet_T
                window_npy = self.get_window_npy(npy_datum, chosen)
                if window_npy is None:
                    break
                window_fnames.append(window_npy)
            if len(window_fnames) != 4:
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

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    d = (d + 1) / 2 # Normalize to [0, 1]
    loss = logloss(d.unsqueeze(1), y)

    return loss

def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step
    
    while global_epoch < nepochs:
        running_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, mel, y) in prog_bar:
            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, global_step, device, model, checkpoint_dir)

            prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)))
        print(f"Global_epoch: {global_epoch}")
        global_epoch += 1

def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
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


# Checkpoint functions should remain the same as in color_syncnet_train.py

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

if __name__ == '__main__':
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup
    test_dataset = Dataset('val')
    # test_dataset[0]
    train_dataset = Dataset('train')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=8)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                        lr=hparams.syncnet_lr)

    print("Loading checkpoint path")
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    print("Begining Training")
    train(device, model, train_data_loader, test_data_loader, optimizer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=hparams.syncnet_checkpoint_interval,
        nepochs=hparams.nepochs)