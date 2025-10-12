import st_gcn_test as st
from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import audio_only
from models import LandmarkSTGCNConformerWithOrientation, LandmarkSTGCNConformer
from models import build_adjacency

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

syncnet_T = 5
syncnet_mel_step_size = 16
ID_LEN = 5 #The number of digits in the id in the file name

video_root = '/home/ksw38/groups/grp_lip/nobackup/autodelete/datasets/fslgroup/grp_lip/compute/datasets/LRS2/preprocessedRetinaface/lrs2/lrs2_video_seg24s/mvlrs_v1/main/'
data_root = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_mp/main/'

audio_model_checkpoint_dir = "ckpt_folder/checkpoints_mediapipe_audio"
st_gcn_checkpoint_dir = "ckpt_folder/checkpoints_mediapipe"
global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))


parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')
parser.add_argument('--video_root', help='Root folder of the videos of the LRS2 dataset', default=video_root)
parser.add_argument("--data_root", help="Root folder of the preprocessed landmarks for LRS2 dataset", default=data_root)
parser.add_argument("--st_gcn_checkpoint_dir", help="Checkpoints for ST GCN", default=st_gcn_checkpoint_dir)
parser.add_argument("--audio_checkpoint_dir", help="Checkpoints for Audio", default=audio_model_checkpoint_dir)
args = parser.parse_args()

class Dataset(object):
    def __init__(self, split, shuffle=True):
        # self.all_videos = get_npy_list(args.data_root, split)
        self.all_videos = get_image_list(args.video_root, split)
        if not shuffle:
            self.order_idx = 0
            self.all_videos.sort()
        self.shuffle = shuffle

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
            if self.shuffle:
                idx = random.randint(0, len(self.all_videos) - 1)
            else:
                self.order_idx += 1
                idx = self.order_idx

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
            npy_path = npy_head + '.npy'

            if not os.path.exists(npy_path):
                continue

            # retrive the data from the npy file
            npy_data = np.load(npy_path)
            num_frames = npy_data.shape[0]

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

                # get the window of npy data from start_id to start_id + syncnet_T
            window_npy = self.get_window_npy(npy_data, chosen)
            if window_npy is None:
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

            x= torch.FloatTensor(window_npy)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            if x is None or mel is None:
                continue

            return x, mel, y

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = F.cosine_similarity(a, v)          # [-1, 1]
    d = (d + 1) / 2                        # [0, 1]
    d = torch.clamp(d, min=0., max=1.)
    d_before = d
    if not torch.equal(d_before, d):
        print(f"Clamped {d_before} to {d}")
    loss = logloss(d.unsqueeze(1), y)      # BCE works safely
    return loss

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

def eval_model(test_data_loader, device, st_gcn_model, audio_model, strip_z = True):
    eval_steps = 1400
    # eval_steps = None ## Modification ##
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):

            st_gcn_model.eval()
            audio_model.eval()

            # Transform data to CUDA device

            mel = mel.to(device)

            x = x.permute(0, 3, 1, 2)
            if strip_z:
                x = x[:, :2, :, :]

            lmk_feat = st_gcn_model(x)
            v = lmk_feat.mean(dim=1)
            a = audio_model(mel)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            losses.append(loss.item())

            if eval_steps is not None and step > eval_steps: break ## Modification ##

        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)

        return

def train(device, st_gcn_model, audio_model, train_data_loader, test_data_loader, st_gcn_optimizer, audio_optimizer,
          st_gcn_checkpoint_dir=None, audio_checkpoint_dir=None, strip_z=True, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step
    
    while global_epoch < nepochs:
        running_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, mel, y) in prog_bar:
            st_gcn_model.train()
            audio_model.train()
            st_gcn_optimizer.zero_grad()
            audio_optimizer.zero_grad()

            # Transform data to CUDA device

            mel = mel.to(device)

            x = x.permute(0, 3, 1, 2)
            if strip_z:
                x = x[:, :2, :, :]

            lmk_feat = st_gcn_model(x)
            v = lmk_feat.mean(dim=1)
            a = audio_model(mel)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            loss.backward()
            st_gcn_optimizer.step()
            audio_optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    st_gcn_model, st_gcn_optimizer, global_step, st_gcn_checkpoint_dir, global_epoch)
                
            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    audio_model, audio_optimizer, global_step, audio_checkpoint_dir, global_epoch)

            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, device, st_gcn_model, audio_model, strip_z=strip_z)

            prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)))
        print(f"Global_epoch: {global_epoch}")
        global_epoch += 1


def _load(checkpoint_path, use_cuda): ## Modification, added use_cuda
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False, use_cuda=False): ## Modification, added use_cuda
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path, use_cuda)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model

def get_checkpoint(checkpoint):
    if os.path.isdir(checkpoint):
        checkpoint_path = os.listdir(checkpoint)[-1]
        checkpoint_path = os.path.join(checkpoint, checkpoint_path)
    else:
        checkpoint_path = checkpoint
    return checkpoint_path

def load_from_checkpoint_or_dir(checkpoint, model, optimizer, reset_optimizer=False, use_cuda=False):
    checkpoint_path = get_checkpoint(checkpoint)
    load_checkpoint(checkpoint_path, model, optimizer=optimizer, reset_optimizer=reset_optimizer, use_cuda=use_cuda)
    return model

def load_stgcn_and_audio_models(checkpoint, audio_checkpoint, A, V, use_cuda = False):
    print(f"Loading LandmarkSTGCNConformer Model from checkpoint {checkpoint}")
    device = torch.device("cuda" if use_cuda else "cpu")
    model_args = dict(
            num_nodes=V,
            A=A,                          # [K, V, V] adjacency
            d_model=128,
            post_linear_hidden=128,       # hidden size before conformer
            conformer_layers=4,
            conformer_heads=4,
            conformer_ff=256,
            conformer_conv_kernel=31
        )
    
    model = LandmarkSTGCNConformer(**model_args)
    model.to(device)

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                            lr=hparams.syncnet_lr, weight_decay=1e-5)
    print('total trainable params for stgcn: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model = load_from_checkpoint_or_dir(checkpoint, model=model, optimizer=optimizer, use_cuda=use_cuda)
    model.eval()

    print(f"\t and audio model from {audio_checkpoint}")
    audio_model = audio_only().to(device)
    audio_optimizer = optim.Adam([p for p in audio_model.parameters() if p.requires_grad],
                                lr=hparams.syncnet_lr, weight_decay=1e-5)
    print(audio_checkpoint)
    audio_model = load_from_checkpoint_or_dir(audio_checkpoint, model=audio_model, optimizer=audio_optimizer, use_cuda=use_cuda)
    audio_model.eval()

    print('total trainable params for stgcn: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('total trainable params for audio: {}'.format(sum(p.numel() for p in audio_model.parameters() if p.requires_grad)))
    return model, audio_model

if __name__=="__main__":

    # use_cuda = torch.cuda.is_available()
    use_cuda = False
    device = "cuda" if use_cuda else "cpu"
    print(f"Using {device}")

    ## Data set-up

    data_limit = 2
    batch_size = hparams.syncnet_batch_size
    num_workers = 1 #8
    test_dataset = Dataset('val') #shuffle = False for debugging purposes
    train_dataset = Dataset('train')
    
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=num_workers)
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers)
    
    
    ## Model set-up
    edges = list(FACEMESH_TESSELATION)
    V = 478
    A = build_adjacency(V, edges)

    model_args = dict(
            num_nodes=V,
            A=A,                          # [K, V, V] adjacency
            d_model=128,
            post_linear_hidden=128,       # hidden size before conformer
            conformer_layers=4,
            conformer_heads=4,
            conformer_ff=256,
            conformer_conv_kernel=31
        )
    
    st_gcn_model = LandmarkSTGCNConformer(**model_args)
    st_gcn_model.to(device)
    st_gcn_optimizer = optim.Adam([p for p in st_gcn_model.parameters() if p.requires_grad],
                            lr=hparams.syncnet_lr, weight_decay=1e-5)

    audio_model = audio_only().to(device)
    audio_optimizer = optim.Adam([p for p in audio_model.parameters() if p.requires_grad],
                            lr=hparams.syncnet_lr, weight_decay=1e-5)
    
    print("Beginning Training")
    train(device, st_gcn_model, audio_model, train_data_loader, test_data_loader, st_gcn_optimizer, audio_optimizer,
          st_gcn_checkpoint_dir, audio_model_checkpoint_dir, checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)
    
    # with torch.no_grad():
    #     # prog_bar = tqdm(enumerate(test_data_loader))
    #     prog_bar = enumerate(test_data_loader)
    #     for step, item in prog_bar:

    #         x, mel, y = item
    #         x = x.permute(0, 3, 1, 2)
    #         x = x[:, :2, :, :] # Strip away z, modify model to accept it for ablation
    #         print(x.shape, mel.shape, y)

    #         st_gcn_model.eval()
    #         v = st_gcn_model(x)

    #         audio_model.eval()
    #         a = audio_model(mel)
    #         print(v.shape, a.shape)

    #         if data_limit is not None and step > data_limit:
    #             break