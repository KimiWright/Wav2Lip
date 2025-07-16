from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_landmarks_gru2 as SyncNet
from models import lmks_only as lmks_only
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

from scipy import signal

# from lmks_activeFrameDetector import SyncNetInstance

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed landmarks for LRS3 VVAD dataset", default='/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_vvadlrs3/main/x_test/')
parser.add_argument('--ground_truth', help="Ground truth folder of the LRS3 VVAD dataset", default='/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_vvadlrs3/main/y_test.npy')
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default='triplets_checkpoints', type=str)
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

def get_files_list(folder_path):
    print(folder_path)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files = sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group(0)))
    return files

def get_digits_from_filename(filename):
    # Extract the digits from the filename using regex
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group(0))
    else:
        return None

def match_digits(digit, filenames):
    # Check if the digit is present in any of the filenames
    for filename in filenames:
        if str(digit) not in filename:
            return False
    return True

class Dataset(object):
    def __init__(self, data_root, ground_truth):
        self.data_root = data_root
        all_files = get_files_list(data_root) 
        self.lmks_files = [f for f in all_files if f.endswith('lmks.npy')]
        self.roll_files = [f for f in all_files if f.endswith('roll.npy')]
        self.pitch_files = [f for f in all_files if f.endswith('pitch.npy')]
        self.yaw_files = [f for f in all_files if f.endswith('yaw.npy')]   

        y = np.load(ground_truth)
        self.y = y    

    def __len__(self):
        return len(self.lmks_files)

    def __getitem__(self, idx):
        # Syncnet is set up randomly sync or not sync a video, that is part of why they take out 5 frame chunks
        while 1:
            # choose a random video
            idx = random.randint(0, len(self.lmks_files) - 1)

            lmks_file = self.lmks_files[idx]
            roll_file = self.roll_files[idx]
            pitch_file = self.pitch_files[idx]
            yaw_file = self.yaw_files[idx]

            npy_file_names = [lmks_file, roll_file, pitch_file, yaw_file]

            if not match_digits(idx, npy_file_names): #Check to make sure the files are from the same video
                continue
            
            npy_files = [os.path.join(self.data_root, f) for f in npy_file_names]

            # retrive the data from the npy files
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
            x_still = np.tile(x_video[0], (num_frames, 1)) # Make a still video of the first frame

            y = self.y[idx]

            return x_video, x_still, y

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    d = (d + 1) / 2 # Normalize to [0, 1]
    loss = logloss(d.unsqueeze(1), y)

    return loss

def eval_model(test_data_loader, device, model):
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    best_accuracies = []
    best_blankThresholds = []
    for step, (x_video, x_still, y) in enumerate(test_data_loader):
        with torch.no_grad():
            x_video = stacker(x_video, device)
            x_still = stacker(x_still, device)
           
            y = y.to(device)

            feat_video = model(x_video)
            # print("Model output shape:", feat_video.shape)
            feat_still = model(x_still)
            # print("Model still output shape:", feat_still.shape)

            fconfm = computeDist(feat_video, feat_still, vshift=15)
            accuracies = []
            max_accuracy = 0
            best_blankThreshold = -1
            for i in np.arange(-1, 1, 0.01):
                blankThreshold = i
                acc = test_accuracy(y, blankThreshold, fconfm, False)
                accuracies.append(acc)
                if acc > max_accuracy:
                    max_accuracy = acc
                    best_blankThreshold = blankThreshold
            print(f"Best accuracy: {max_accuracy:.3f} at threshold {best_blankThreshold:.3f}")
            best_accuracies.append(max_accuracy)
            best_blankThresholds.append(best_blankThreshold)

    print("Average Accurracy", np.average(best_accuracies))
    print("Average blankThreshold", np.average(best_blankThresholds))
    print("Best accuracies:", best_accuracies)
    print("Best blank thresholds:", best_blankThresholds)

def eval_model_batch_size_1(test_data_loader, device, model):
    print("Evaluating model with batch size 1")
    accuracies = []
    blankThresholds = []
    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []
    ys = []
    best_accuracy = 0
    best_blankThreshold = -1
    with torch.no_grad():
        for i in np.arange(-1, 1, 0.01):
            for step, (x_video, x_still, y) in enumerate(test_data_loader):
                x_video = x_video.to(torch.float32).to(device)
                x_still = x_still.to(torch.float32).to(device)
                y = y.to(device)
                feat_video = model(x_video)
                feat_still = model(x_still)

                # offset, conf, dists_npy, fconfm = computeDist(feat_video, feat_still, vshift=15)
                fconfm = computeDist(feat_video, feat_still, vshift=15)
                blankThreshold = i
                result = fconfm.item() < blankThreshold # >
                if result and y.item() == 1:
                    true_positives.append(fconfm.item())
                elif result and y.item() == 0:
                    false_positives.append(fconfm.item())
                elif not result and y.item() == 1:
                    false_negatives.append(fconfm.item())
                elif not result and y.item() == 0:
                    true_negatives.append(fconfm.item())
                ys.append(y.item())
            ys_array = np.array(ys)
            true_positive_rate = len(true_positives) / sum(ys_array == 1)
            false_positive_rate = len(false_positives) / sum(ys_array == 0)
            true_negative_rate = len(true_negatives) / sum(ys_array == 0)
            false_negative_rate = len(false_negatives) / sum(ys_array == 1)
            accuracy = ((true_positive_rate * sum(ys_array == 1)) + (true_negative_rate * sum(ys_array == 0))) / len(ys_array)
            accuracies.append(accuracy)
            blankThresholds.append(blankThreshold)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_blankThreshold = blankThreshold
    print(f"Best accuracy: {best_accuracy:.3f} at threshold {best_blankThreshold:.3f}")

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

def collate_variable_length(batch):
    x_v_batch = [item[0] for item in batch]  # list of (seq_len, 187)
    x_s_batch = [item[1] for item in batch]  # list of (seq_len, 187)
    y_batch = torch.tensor([item[2] for item in batch], dtype=torch.bool)
    
    return x_v_batch, x_s_batch, y_batch

def stacker(batch, device):
    ## Temporary fix, only use the first 5 frames so things can be evaluated
    ## This is how SyncNet is set up, it takes 5 frames at a time, anyway
    first_5_frames = [t[:5] for t in batch]  # each is now (5, 187)
    stacked = torch.Tensor(np.stack(first_5_frames)) 
    stacked_batch = stacked.to(device)  # (B, 5, 187)
    return stacked_batch

def calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift*2+1
    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))
    dists = []
    for i in range(0,len(feat1)):
        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))
    return dists

def computeDist(feat1, feat2, vshift=15):
    dists = calc_pdist(feat1, feat2, vshift=vshift)
    mdist = torch.mean(torch.stack(dists, 1), 1)
    minval, minidx = torch.min(mdist, 0)

    mdist = mdist.detach().cpu()
    minidx = minidx.item()

    fdist = np.stack([dist[minidx].detach().cpu().numpy() for dist in dists])
    fconf = torch.median(mdist).item() - fdist
    if fconf.shape[0] < 9:
        kernel = fconf.shape[0] // 2 * 2 + 1  # Next odd number below size
    else:
        kernel = 9
    fconfm = signal.medfilt(fconf, kernel_size=kernel)


    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    return fconfm

def test_accuracy(y, blankThreshold, fconfm, print_results=True):
    true_positive_rate = sum(fconfm[y == 1] < blankThreshold) / sum(y == 1)
    false_positive_rate = sum(fconfm[y == 0] < blankThreshold) / sum(y == 0)
    true_negative_rate = sum(fconfm[y == 0] >= blankThreshold) / sum(y == 0)
    false_negative_rate = sum(fconfm[y == 1] >= blankThreshold) / sum(y == 1)
    accuracy = ((true_positive_rate * sum(y == 1)) + (true_negative_rate * sum(y == 0))) / len(y)
    if print_results:
        print(f"Blank Threshold: {blankThreshold:.3f}, Accuracy: {accuracy:.3f}")
        print(f"True Positive Rate: {true_positive_rate:.3f}, False Positive Rate: {false_positive_rate:.3f}")
        print(f"True Negative Rate: {true_negative_rate:.3f}, False Negative Rate: {false_negative_rate:.3f}")
        print()
    return accuracy

def load_face_model(checkpoint_path, device, startswith='face'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    full_state_dict = checkpoint['state_dict']

    face_state_dict = {k: v for k, v in full_state_dict.items() if k.startswith("face")}

    model = lmks_only().to(device)
    missing, unexpected = model.load_state_dict(face_state_dict, strict=False)
    if missing:
        print("Missing keys in the state_dict:", missing)
    if unexpected:
        print("Unexpected keys in the state_dict:", unexpected)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    return model

#  python landmarks_syncnet_eval.py --checkpoint_path "/home/ksw38/RVL/color_syncnet/Wav2Lip/pre_loaded_audio_checkpoints/checkpoint_step000960000.pth"
if __name__ == '__main__':
    # Dataset and Dataloader setup
    test_dataset = Dataset(args.data_root, args.ground_truth)

    device = torch.device("cuda" if use_cuda else "cpu")
    # model = SyncNet().to(device)
    # print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    # If no checkpoint path is provided, load the latest checkpoint from the directory
    if checkpoint_path is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    model = load_face_model(checkpoint_path, device)

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)
    
    # test_data_loader = data_utils.DataLoader(
    # test_dataset, batch_size=hparams.syncnet_batch_size,
    # num_workers=8, collate_fn=collate_variable_length)

    # eval_model(test_data_loader, device, model)

    test_data_loader_batch_size_1 = data_utils.DataLoader(
    test_dataset, batch_size=1,
    num_workers=8)

    eval_model_batch_size_1(test_data_loader_batch_size_1, device, model)
            
    # s = SyncNetInstance()
    # s.loadParameters(checkpoint_path, use_cuda=use_cuda)
    # s.eval()

    
    