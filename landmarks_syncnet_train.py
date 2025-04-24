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
from hparams import hparams

from collections import defaultdict
from os import path

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", default='/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main/')

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default='landmarks_checkpoints', type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)

args = parser.parse_args()


global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

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
    def __init__(self, split): # So far only edited function
        print(f"init data from {args.data_root}")
        print(args.data_root)
        self.all_videos = get_npy_list(args.data_root, split)
        print(self.all_videos[0])

    def get_frame_id(self, frame):
        print("get_frame_id")
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        print("get_window")
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        print("crop_audio_window")
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]


    def __len__(self):
        print("len")
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

    # print(data)

    test_dataset = Dataset('val')

    # test_data_loader = data_utils.DataLoader(
    #     test_dataset, batch_size=1, #hparams.syncnet_batch_size,
    #     num_workers=8)
    # print("Test Dataloader")

    # first_batch = next(iter(test_data_loader))
    # print("first_batch")

    # (x, mel, y) = first_batch