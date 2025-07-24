import os
import re
import numpy as np
from tqdm import tqdm

import torch
from torch import optim
import torch.utils.data as data_utils

from hparams import hparams
from models.lmks_only import lmks_only
from models.audio_only import audio_only
import landmarks_syncnet_train_gru2 as gru2

data_root = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_vvadlrs3/main/x_test/'
ground_truth = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_vvadlrs3/main/y_test.npy'
train_data_root = '/home/ksw38/groups/grp_landmarks/nobackup/autodelete/landmarks_vvadlrs3/main/x_train/'
ground_truth_train = '/home/ksw38/groups/grp_landmarks/nobackup/autodelete/landmarks_vvadlrs3/main/y_train.npy'

checkpoint_dir = 'landmarks_checkpoints_gru2'
checkpoint_dir = "triplets_checkpoints"
checkpoint_path = None


################################
# Get Data and support functions
################################

def get_files_list(folder_path):
    print(folder_path)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files = sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group(0)))
    return files

def match_digits(digit, filenames):
    # Check if the digit is present in any of the filenames
    for filename in filenames:
        if str(digit) not in filename:
            return False
    return True

def get_window_npy(data, syncnet_T = 5, start_id=0):
    if start_id + syncnet_T < len(data):
        return data[start_id : start_id + syncnet_T]
    else:
        return None

def get_data(data_root, ground_truth, data_point_limit=None, start_idx=0):
    all_files = get_files_list(data_root) 
    lmks_files = [f for f in all_files if f.endswith('lmks.npy')]
    roll_files = [f for f in all_files if f.endswith('roll.npy')]
    pitch_files = [f for f in all_files if f.endswith('pitch.npy')]
    yaw_files = [f for f in all_files if f.endswith('yaw.npy')]   

    y = np.load(ground_truth)
    
    data = []
    for idx in tqdm(range(len(lmks_files))):
        if idx < start_idx:
            continue

        lmks_file = lmks_files[idx]
        roll_file = roll_files[idx]
        pitch_file = pitch_files[idx]
        yaw_file = yaw_files[idx]

        npy_file_names = [lmks_file, roll_file, pitch_file, yaw_file]

        if not match_digits(idx, npy_file_names):
            continue
        npy_files = [os.path.join(data_root, f) for f in npy_file_names]
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

        min_frames = 5 # minimum number of frames for the kernel size
        if x_video.shape[0] < min_frames: 
            # print(f"Video too short for kernel size {kernel_size}: {x_video.shape[0]}")
            continue  # Skip if the video is too short for the kernel size
        
        x_video_and_rot = (x_video, x_roll, x_pitch, x_yaw)

        data.append((x_video_and_rot, y[idx]))
        if data_point_limit is not None and len(data) >= data_point_limit:
            break

    return data

################
# Datasets
#################
    
class Dataset_5_Frame_Rotation(object):
    def __init__(self, split = 'test'):
        if split == 'test':
            self.data = get_data(data_root, ground_truth)
        elif split == 'train':
            self.data = get_data(train_data_root, ground_truth)
        else:
            raise ValueError("Split must be 'test' or 'train'")
        
        self.processed_data = []
        for datum in self.data:
            x_tuple, y = datum
            x_video_full, x_roll, x_pitch, x_yaw = x_tuple
            if x_video_full is None:
                raise ValueError("x_video_full is None")
            x_video = get_window_npy(x_video_full, start_id=0)
            if x_video is not None:
                self.processed_data.append(((x_video, x_roll, x_pitch, x_yaw), y))
            else:
                print(f"Skipping data point {x_video_full.shape} due to insufficient frames")

    def __len__(self):
        return len(self.processed_data)
    def __getitem__(self, idx):
        return self.processed_data[idx]


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


babble_embedding_path = "/home/ksw38/RVL/color_syncnet/Wav2Lip/kimi/babble_embedding.npy"
babble_emb = torch.Tensor(np.load(babble_embedding_path))
def babble_loop(model, data_loader, device): # Try the loss from contrastive learning
    with torch.no_grad():
        losses = []
        y_vals = []
        global babble_emb
        babble_emb = babble_emb.to(device)
        roll_max = 0
        roll_min = 0
        for step, (x, y) in enumerate(data_loader):
            x_video, x_roll, x_pitch, x_yaw = x
            # Change shape from [1, Frames, 1] to [Frames]
            x_roll = x_roll.squeeze()
            if max(x_roll) > roll_max:
                roll_max = max(x_roll)
            if min(x_roll) < roll_min:
                roll_min = min(x_roll)

            # print(max(x_roll), max(x_pitch), max(x_yaw))
            # print(min(x_roll), min(x_pitch), min(x_yaw))
            x_video = x_video.to(device).to(torch.float32)
            x_roll = x_roll.to(device).to(torch.float32)
            x_pitch = x_pitch.to(device).to(torch.float32)
            x_yaw = x_yaw.to(device).to(torch.float32)
  
            y = y.to(device).to(torch.float32).unsqueeze(0)
            y_vals.append(int(y.item()))
            v = model(x_video)
            loss = gru2.cosine_loss(babble_emb, v, y)
            losses.append(loss.cpu().item())
        print(f"Roll range: {roll_min} to {roll_max}")
        return losses, y_vals

####################
# Accuracy functions
####################

def accuracy(ys, test_ys):
    correct = 0
    for y, test_y in zip(ys, test_ys):
        if y == test_y:
            correct += 1
    return correct / len(ys)

def test_accuracy(losses, true_y, threshold, flip=False):
    if flip:
        results = [1.0 if loss < threshold else 0.0 for loss in losses]
    else:
        results = [0.0 if loss < threshold else 1.0 for loss in losses]
    acc = accuracy(true_y, results)
    return acc

if __name__ == "__main__":
    device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shuffle_dataset = False
    num_workers = 1
    batch_size = 1
    threshold = 0.72 # Threshold for accuracy

    print("Loading checkpoint path")
    if checkpoint_path  is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    lmks_model = load_partial_model(checkpoint_path, device=device, startswith='face')
    lmks_model.eval()

    test_dataset = Dataset_5_Frame_Rotation('test')
    print(f"\nNumber of samples in 5-frame chunks dataset: {len(test_dataset)}")

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=8, drop_last=True, shuffle=shuffle_dataset)
    
    losses, y_vals = babble_loop(lmks_model, test_data_loader, device)
    acc_test = test_accuracy(losses, y_vals, threshold, flip=False)
    print(f"Test accuracy: {acc_test}")
