from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
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
#                                                                                                  /home/ksw38/groups/grp_lip/nobackup/autodelete/datasets/fslgroup/grp_lip/compute/datasets/LRS2/preprocessedRetinaface/lrs2/lrs2_video_seg24s/mvlrs_v1/main
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default='landmarks_checkpoints', type=str)
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

    def get_window(self, start_frame):
        # In the current iteration of get_item, DON'T USE THIS FUNCTION
        # This function gets a window of frames from a video, the landmarks, however, are not set up a isolated jpg files, 
        # but as a numpy file with the landmarks for all frames in the video
        # Get a window of frames from the start_frame
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = [] # lmks, roll, pitch, yaw window fnames
        window_fnames_lmks = []
        window_fnames_roll = []
        window_fnames_pitch = []
        window_fnames_yaw = []
        # Get the window of frames in the range of start_id to start_id + syncnet_T
        for frame_id in range(start_id, start_id + syncnet_T):
            frame_id_str = str(frame_id).zfill(ID_LEN)
            frame_lmks = join(vidname, frame_id_str + '_lmks.npy')
            frame_roll = join(vidname, frame_id_str + '_roll.npy')
            frame_pitch = join(vidname, frame_id_str + '_pitch.npy')
            frame_yaw = join(vidname, frame_id_str + '_yaw.npy')
            # If the window doesn't contain syncnet_T frames, return None
            if not isfile(frame_lmks) or not isfile(frame_roll) or not isfile(frame_pitch) or not isfile(frame_yaw):
                return None
            window_fnames_lmks.append(frame_lmks)
            window_fnames_roll.append(frame_roll)
            window_fnames_pitch.append(frame_pitch)
            window_fnames_yaw.append(frame_yaw)
        # Combine lmks, roll, pitch, yaw window fnames into a list
        window_fnames.append(window_fnames_lmks)
        window_fnames.append(window_fnames_roll)
        window_fnames.append(window_fnames_pitch)
        window_fnames.append(window_fnames_yaw)
        return window_fnames

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
                    continue
                window_fnames.append(window_npy)

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
            
            # Reshape and concatenate the npy data ## May want to look at different ways to concatenate the data to better preserve temporal information
            x_lmks = window_fnames[0].reshape(syncnet_T, -1)
            x_roll = window_fnames[1][:, None]
            x_pitch = window_fnames[2][:, None]
            x_yaw = window_fnames[3][:, None]
            x = np.concatenate([x_lmks, x_roll, x_pitch, x_yaw], axis=1)

            # # H x W x 3 * T
            # x = np.concatenate(window, axis=2) / 255.
            # x = x.transpose(2, 0, 1)
            # x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y

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
    test_dataset = Dataset('val')
    test_file = test_dataset.all_videos[20]

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=1, #hparams.syncnet_batch_size,
        num_workers=8)
    print("Test Dataloader")

    first_batch = next(iter(test_data_loader))
    print("first_batch")

    (x, mel, y) = first_batch
    print("x shape: ", x.shape)
    print("mel shape: ", mel.shape)
    print("y: ", y)