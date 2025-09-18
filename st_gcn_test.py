import models.st_gcn as model
# import run_statistics as run_stats

from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import audio_color as SyncNet
from models.audio_only import audio_only
import load_model
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
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION 

## Variables ##

syncnet_T = 5
syncnet_mel_step_size = 16
ID_LEN = 5 #The number of digits in the id in the file name

video_root = '/home/ksw38/groups/grp_lip/nobackup/autodelete/datasets/fslgroup/grp_lip/compute/datasets/LRS2/preprocessedRetinaface/lrs2/lrs2_video_seg24s/mvlrs_v1/main/'
data_root = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_preprocessed/main'
data_root = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main'

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')
parser.add_argument('--video_root', help='Root folder of the videos of the LRS2 dataset', default=video_root)
parser.add_argument("--data_root", help="Root folder of the preprocessed landmarks for LRS2 dataset", default=data_root)
args = parser.parse_args()

## Dataset ##

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
            x_lmks = window_fnames[0]
            x_roll = window_fnames[1]
            x_pitch = window_fnames[2]
            x_yaw = window_fnames[3]
            # x = np.concatenate([x_lmks, x_roll, x_pitch, x_yaw], axis=1)

            x_rot = np.vstack((x_roll, x_pitch, x_yaw))
            
            x = torch.FloatTensor(x_lmks)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, x_rot, mel, y

## Edges ##

def knn_edges(points_xy, k=4):
    """
    Build undirected edges by connecting each landmark
    to its k nearest neighbors.
    
    Args:
        points_xy: np.array [V, 2], landmark coordinates (x,y)
        k: number of neighbors to connect
    
    Returns:
        edges: list of (i, j) tuples
    """
    V = points_xy.shape[0]
    edges = set()
    for i, p in enumerate(points_xy):
        # distances from point i to all others
        dists = np.linalg.norm(points_xy - p, axis=1)
        # get indices of k nearest (skip self at index 0)
        nearest = np.argsort(dists)[1:k+1]
        for j in nearest:
            edges.add((i, j))
            edges.add((j, i))  # make undirected
    return list(edges)

if __name__ == "__main__":
    data_limit = 4
    batch_size = 1 # hparams.syncnet_batch_size
    test_dataset = Dataset('val')
    # use_cuda = torch.cuda.is_available()
    use_cuda = False
    device = "cuda" if use_cuda else "cpu"
    print(f"Using {device}")

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=8)
    


    ## Generate Edges and Adjacency Matrix, maybe adjust later ##
    first_point = test_dataset[0]
    (x, x_rot, mel, y) = first_point
    first_lmks = x[0].T
    edges = knn_edges(first_lmks)
    num_lmks = first_lmks.shape[0]
    # edges = list(FACEMESH_TESSELATION)
    # print(edges)
    # print(f"len(edges) {len(edges)}") # Use these in mediapipe version

    print(num_lmks)

    A = model.build_adjacency(num_lmks, edges)
    V = num_lmks
    C = 2 # x,y maybe updata to 5 include roll pitch yaw? but those are frame-wise, so maybe I can include it somewhere else?
    K = 1 # 1 partion, chosen arbitarily 
    

    temporal_kernel_size = 9
    stgcn_dropout = 0.0

    ## Init model
    print("Loading LandmarkSTGCNConformer Model")
    model = model.LandmarkSTGCNConformer(
        num_nodes=V,
        A=A,
        d_model=128,
        post_linear_hidden=128,
        conformer_layers=4,
        conformer_heads=4,
        conformer_ff=256,
        conformer_conv_kernel=31
    )
    model.to(device)
    print('total trainable params for stgcn: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model.eval()

    # print("Loading SyncNet Model")
    # checkpoint_path = "/home/ksw38/RVL/color_syncnet/Wav2Lip/lipsync_expert.pth"
    # syncnet = SyncNet().to(device)
    # print('total trainable params {}'.format(sum(p.numel() for p in syncnet.parameters() if p.requires_grad)))
    print("Loading Audio only Model")
    audio_model = audio_only().to(device)
    optimizer = optim.Adam([p for p in audio_model.parameters() if p.requires_grad],
                            lr=hparams.syncnet_lr, weight_decay=1e-5)
    # load_model.load_checkpoint(checkpoint_path, syncnet, optimizer, False, use_cuda)
    
    audio_checkpoint_path = "/home/ksw38/RVL/color_syncnet/Wav2Lip/landmarks_checkpoints_gru2/checkpoint_step001800000.pth"
    load_model.load_partial_model(checkpoint_path=audio_checkpoint_path, device=device, startswith='audio')
    audio_model.eval()
    

    ## Loop ##
    with torch.no_grad():
        # prog_bar = tqdm(enumerate(test_data_loader))
        prog_bar = enumerate(test_data_loader)
        for step, (x, x_rot, mel, y) in prog_bar:
            print(f"Step {step}")
            print(x.shape, x_rot.shape, mel.shape, y)
            x = x.permute(0, 2, 1, 3)

            lmk_feat = model(x)
            v = lmk_feat.mean(dim=1)
            a = audio_model(mel)

            print(v.shape, a.shape)
            sim = F.cosine_similarity(a, v)
            print(sim)


            if data_limit is not None and step > data_limit:
                break