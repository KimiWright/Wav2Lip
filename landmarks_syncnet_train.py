from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
import audio

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
# So the stradegy in the color_syncnet_train.py is to use the name of the mp4 files as the id and find the correspoinding mel and image files
# I flipped the stradegy in the landmarks_syncnet_train.py to use the name of the landmarks files
# I'm trying to decide if this is a good idea or not
# Probably not, I want every frame in an mp4 to be in the same batch, but they aren't the way I have it set up now

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

    def get_window(self, start_frame):
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

    def crop_audio_window(self, spec, start_frame):
        print("crop_audio_window")
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        print("getitem")
        while 1:
            # print(self.all_videos)
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]
            vidname_no_ext = os.path.splitext(vidname)[0]

            vidname_file = os.path.splitext(os.path.basename(vidname))[0]
            vidname_folder = os.path.basename(os.path.dirname(vidname))
            vidname_loc = join(args.data_root, vidname_folder, vidname_file)
            img_names = list(glob(join(vidname_loc, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                print("len")
                continue
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                print("window_fnames is None")
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read: continue

            try:
                wavpath = vidname_no_ext + ".wav"
                print(wavpath)
                print(vidname)
                print()
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                print("exception e")
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                print("mel")
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            print("Got Item")
            return x, mel, y
        
if __name__ == '__main__':
    # load_path = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main/5535415699068794046/00001_lmks.npy"
    # data = np.load(load_path)

    test_dataset = Dataset('val')
    test_file = test_dataset.all_videos[20]
    # print(test_file)
    test_npy = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main/5535415699068794046/00001_lmks.npy"
    test_npy = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main/5535415699068794046/00002_lmks.npy"
    test_npy = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main/5535496873950688380/00019_lmks.npy"

    print(test_dataset.get_window(test_npy)) # Doesn't work partly because it is the wrong file, look at _get_item_
    # Example output in color_syncnet_train.py Window_fnames:  ['lrs2_preprocessed/6331559613336179781/00019/2.jpg', 'lrs2_preprocessed/6331559613336179781/00019/3.jpg', 'lrs2_preprocessed/6331559613336179781/00019/4.jpg', 'lrs2_preprocessed/6331559613336179781/00019/5.jpg', 'lrs2_preprocessed/6331559613336179781/00019/6.jpg']
    # Exmample files input: lrs2_preprocessed/6227471510414418277/00043/0.jpg, lrs2_preprocessed/6090505003344967660/00046/34.jpg, lrs2_preprocessed/6131718650523849495/00024/2.jpg
    
    #print(test_dataset[0])

    # test_data_loader = data_utils.DataLoader(
    #     test_dataset, batch_size=1, #hparams.syncnet_batch_size,
    #     num_workers=8)
    # print("Test Dataloader")

    # first_batch = next(iter(test_data_loader))
    # print("first_batch")

    # (x, mel, y) = first_batch