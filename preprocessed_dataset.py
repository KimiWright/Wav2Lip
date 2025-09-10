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
import pandas as pd

from glob import glob

import os, random, cv2, argparse
from hparams import hparams

data_preprocessed = "/home/ksw38/groups/grp_landmarks/nobackup/autodelete/AVA_activespeaker/preprocessed/clips"
data_raw = "/home/ksw38/groups/grp_landmarks/nobackup/autodelete/AVA_activespeaker/clips"
labels_csv = "/home/ksw38/groups/grp_landmarks/nobackup/autodelete/AVA_activespeaker/labels_clips.csv"

syncnet_T = 5
syncnet_mel_step_size = 16

def get_image_list(data_root, split=None):
    filelist = os.listdir(data_root)
    split_point = int(len(filelist) * .2) # 20/80 data split
    if split == 'val':
        filelist = filelist[:split_point]
    elif split == 'train':
        filelist = filelist[split_point:] # For benchmarking version remove split function, this is all the offical val set, making split = None should work
        
    return filelist

class Dataset(object):
    def __init__(self, split, data_raw=data_raw, data_preprocessed=data_preprocessed, labels_csv=labels_csv):
        self.data_raw = data_raw
        self.data_preprocessed = data_preprocessed
        self.all_videos = get_image_list(self.data_raw, split)
        self.df = pd.read_csv(labels_csv)


    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
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
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            data_root = self.data_preprocessed
            idx = random.randint(0, len(self.all_videos) - 1)
            
            vidname = self.all_videos[idx]

            vidname_no_ext = os.path.splitext(vidname)[0]
            vidname_file = os.path.splitext(os.path.basename(vidname))[0]
            vidname_folder = os.path.basename(os.path.dirname(vidname))
            img_names = list(glob(join(data_root, vidname_folder, vidname_file, '*.jpg')))

            if len(img_names) <= 3 * syncnet_T:
                continue
            img_name = random.choice(img_names)

            # indices = self.df.index[self.df["File"] == vidname_file].tolist()
            indices = self.df.index[self.df["File"].str.contains(vidname_file, na=False)]

            if len(indices) == 0:
                continue
            file_idx = indices[0]
            label = self.df["Label"][file_idx]
            # print(idx)
            # print(vidname)
            # print(file_idx)
            # print(label)
            if label == "NOT_SPEAKING":
                y = torch.zeros(1).float()
            else:
                y = torch.ones(1).float()

            chosen = img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
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
                wavpath = join(data_root, vidname_no_ext, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)
                orig_mel = audio.melspectrogram(wav).T ## This only works with the color_syncnet env
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y
        
if __name__ == "__main__":
    test_dataset = Dataset('val')
    # output = test_dataset[0]

    # print(output)

    batch_size = hparams.syncnet_batch_size
    num_workers = 8
    batch_size = 1
    num_workers = 1
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=num_workers)
    
    for step, (x, mel, y) in enumerate(test_data_loader):
        print(f"step {step}")
        print(x.shape, mel.shape, y)

        if step >= 30:
            break