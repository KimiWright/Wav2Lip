from models.st_gcn import LandmarkSTGCNConformer, LandmarkSTGCNConformerWithOrientation, build_adjacency
from models.audio_only import audio_only
import landmarks_audio as audio
from hparams import hparams, get_image_list

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

import numpy as np
import math
import os, random, cv2, argparse
from os.path import dirname, join, basename, isfile
from os import path
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import re
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION 
# from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score

## Variables ##

syncnet_T = 5
syncnet_mel_step_size = 16
ID_LEN = 5 #The number of digits in the id in the file name

video_root = '/home/ksw38/groups/grp_lip/nobackup/autodelete/datasets/fslgroup/grp_lip/compute/datasets/LRS2/preprocessedRetinaface/lrs2/lrs2_video_seg24s/mvlrs_v1/main/'
data_root = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_preprocessed/main'
data_root = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main'
data_root = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_norm/main"

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

            x_rot = torch.FloatTensor(np.vstack((x_roll, x_pitch, x_yaw)))
            
            x_lmks = np.swapaxes(x_lmks, 1,2)
            x = torch.FloatTensor(x_lmks)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, x_rot, mel, y

## Model Loading ##

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

def load_stgcn_and_audio_models(checkpoint, audio_checkpoint, A, V, use_cuda = False, rotation = False):
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
    
    if rotation:
        model = LandmarkSTGCNConformerWithOrientation(**model_args)
    else:
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
    audio_model = load_from_checkpoint_or_dir(audio_checkpoint, model=audio_model, optimizer=audio_optimizer, use_cuda=use_cuda)
    audio_model.eval()

    print('total trainable params for stgcn: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('total trainable params for audio: {}'.format(sum(p.numel() for p in audio_model.parameters() if p.requires_grad)))
    return model, audio_model

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

RVL_FACEMESH_LEFT_EYEBROW = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
RVL_FACEMESH_LEFT_EYE = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
RVL_FACEMESH_LIPS = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
RVL_FACEMESH_RIGHT_EYE = [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]
RVL_FACEMESH_RIGHT_EYEBROW = [82, 83, 84, 85, 86, 87, 88, 89, 90, 91]

def facial_edges():
    edges = []
    for region in [RVL_FACEMESH_LEFT_EYE, RVL_FACEMESH_RIGHT_EYE,
               RVL_FACEMESH_LEFT_EYEBROW, RVL_FACEMESH_RIGHT_EYEBROW,
               RVL_FACEMESH_LIPS]:
        for i in range(len(region)-1):
            edges.append((region[i], region[i+1]))
        edges.append((region[-1], region[0])) 

    return edges

## Miscellanous ##

# def best_accuracy(y_test, y_scores, thresholds):
#     accuracies = []
#     for thr in thresholds:
#         preds = (y_scores >= thr).astype(int)
#         acc = accuracy_score(y_test, preds)
#         accuracies.append(acc)

#     best_acc_idx = max(range(len(accuracies)), key=lambda i: accuracies[i])
#     best_acc_threshold = thresholds[best_acc_idx]
#     best_acc = accuracies[best_acc_idx]

    
#     return best_acc_threshold, best_acc

# def best_f1_score(y_test, y_scores, thresholds):
#     f1s = []
#     for thr in thresholds:
#         preds = (y_scores >= thr).astype(int)
#         f1s.append(f1_score(y_test, preds))

#     best_f1_idx = max(range(len(f1s)), key=lambda i: f1s[i])
#     best_f1_threshold = thresholds[best_f1_idx]
#     best_f1 = f1s[best_f1_idx]

#     return best_f1_threshold, best_f1

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
    # edges = facial_edges()
    num_lmks = first_lmks.shape[0]
    # edges = list(FACEMESH_TESSELATION)
    # print(edges)
    # print(f"len(edges) {len(edges)}") # Use these in mediapipe version

    print(num_lmks)

    A = build_adjacency(num_lmks, edges)
    V = num_lmks
    C = 2 # x,y maybe updata to 5 include roll pitch yaw? but those are frame-wise, so maybe I can include it somewhere else?
    K = 1 # 1 partion, chosen arbitarily 
    

    temporal_kernel_size = 9
    stgcn_dropout = 0.0

    ## Init model
    print("Loading LandmarkSTGCNConformer Model")
    model = LandmarkSTGCNConformer(
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
    st_gcn_optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                            lr=hparams.syncnet_lr, weight_decay=1e-5)
    print('total trainable params for stgcn: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    st_gcn_norot_checkpoint_path = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints_st_gcn_norot/checkpoint_step000050000.pth"
    # st_gcn_norot_checkpoint_path = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints_audio_norot/checkpoint_step000250000.pth"
    model = load_checkpoint(st_gcn_norot_checkpoint_path, model=model, optimizer=st_gcn_optimizer, use_cuda=use_cuda)
    model.eval()

    model_rot = LandmarkSTGCNConformerWithOrientation(
        num_nodes=V,
        A=A,                # [K, V, V] adjacency
        d_model=128,
        post_linear_hidden=128,  # hidden size before conformer
        conformer_layers=4,
        conformer_heads=4,
        conformer_ff=256,
        conformer_conv_kernel=31
    )
    model_rot.to(device)
    print('total trainable params for stgcn with rotation: {}'.format(sum(p.numel() for p in model_rot.parameters() if p.requires_grad)))
    model_rot.eval()

    # print("Loading SyncNet Model")
    # checkpoint_path = "/home/ksw38/RVL/color_syncnet/Wav2Lip/lipsync_expert.pth"
    # syncnet = SyncNet().to(device)
    # print('total trainable params {}'.format(sum(p.numel() for p in syncnet.parameters() if p.requires_grad)))
    print("Loading Audio only Model")
    audio_model = audio_only().to(device)
    optimizer = optim.Adam([p for p in audio_model.parameters() if p.requires_grad],
                            lr=hparams.syncnet_lr, weight_decay=1e-5)
    # lm.load_checkpoint(checkpoint_path, syncnet, optimizer, False, use_cuda)
    
    # audio_checkpoint_path = "/home/ksw38/RVL/color_syncnet/Wav2Lip/landmarks_checkpoints_gru2/checkpoint_step001800000.pth"
    # lm.load_partial_model(checkpoint_path=audio_checkpoint_path, device=device, startswith='audio')
    audio_checkpoint_path = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints_audio_norot/checkpoint_step000050000.pth"
    audio_model = load_checkpoint(audio_checkpoint_path, model=audio_model, optimizer=optimizer, use_cuda=use_cuda)
    audio_model.eval()
    
    sim_vals = []
    ## Loop ##
    with torch.no_grad():
        # prog_bar = tqdm(enumerate(test_data_loader))
        prog_bar = enumerate(test_data_loader)
        for step, (x, x_rot, mel, y) in prog_bar:
            # print(f"Step {step}")
            print(x.shape, x_rot.shape, mel.shape, y)
            x = x.permute(0, 2, 1, 3)

            lmk_feat = model(x)
            v = lmk_feat.mean(dim=1)
            a = audio_model(mel)

            # print(v.shape, a.shape)
            sim = F.cosine_similarity(a, v)
            # print(sim)
            sim_vals.append(sim)

            x_rot = x_rot.permute(0, 2, 1)
            rot_feat = model_rot(x, x_rot)
            # print(rot_feat.shape)


            if data_limit is not None and step > data_limit:
                break

        print(sim_vals)