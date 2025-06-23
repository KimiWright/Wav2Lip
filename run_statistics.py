import os
import re
import numpy as np

import torch
from torch import optim
import torch.utils.data as data_utils

from hparams import hparams
import landmarks_audio as audio
import landmarks_syncnet_train_gru2 as gru2
from models import SyncNet_landmarks_gru2 as SyncNet # Eventually switch to face only and pregenerated audio


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

def get_data(data_root, ground_truth):
    all_files = get_files_list(data_root) 
    lmks_files = [f for f in all_files if f.endswith('lmks.npy')]
    roll_files = [f for f in all_files if f.endswith('roll.npy')]
    pitch_files = [f for f in all_files if f.endswith('pitch.npy')]
    yaw_files = [f for f in all_files if f.endswith('yaw.npy')]   

    y = np.load(ground_truth)
    
    data = []
    for idx in range(len(lmks_files)):
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
                print(f"Error loading npy file {npy_file}: {e}")
                break
        if len(npy_data) != 4:
            continue
        # Check if the data is empty
        if any(data.size == 0 for data in npy_data):
            print(f"Empty data in npy files: {npy_file_names}")
            continue
        
        num_frames = npy_data[0].shape[0]
            
        x_lmks = npy_data[0].reshape(num_frames, -1)
        x_roll = npy_data[1][:, None]
        x_pitch = npy_data[2][:, None]
        x_yaw = npy_data[3][:, None]
        x_video = np.concatenate([x_lmks, x_roll, x_pitch, x_yaw], axis=1)
        data.append((x_video, y[idx]))

    return data

################
# Datasets
#################

data_out = get_data(data_root, ground_truth)

class Dataset_Full_Video(object):
    def __init__(self, data_root, ground_truth):
        self.data = data_out

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x_video, y = self.data[idx]
        return x_video, y
    
class Dataset_5_Frame(object):
    def __init__(self, data_root, ground_truth):
        self.data = data_out

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x_video, y = self.data[idx]
        x_video = get_window_npy(x_video, start_id=0)
        return x_video, y
    
def generate_mel_for_frames(num_frames, silence = True, video_fps=hparams.fps, mel_fps=80, sample_rate=16000, hop_length=200):
    mel_frames = int(num_frames * mel_fps / video_fps)
    num_samples = (mel_frames - 1) * hop_length  # +1 mel frame per hop
    if silence:
        gen_audio = torch.zeros(num_samples)
    else:
        gen_audio = torch.randn(num_samples) # Generate white noise
    # Compute mel spectrogram
    mel = audio.melspectrogram(gen_audio).T  # [Time, Mel]
    mel = mel[:mel_frames]  # Clip to exact mel_frames
    mel = torch.FloatTensor(mel.T).unsqueeze(0)  # [1, 80, mel_frames]
    return mel

sound_types = ["silence", "white_noise"] # babble_noise

def audio_loop(model, data_loader): # Try the loss from contrastive learning
    with torch.no_grad():
        num_sounds = len(sound_types)
        losses = [[] for _ in range(num_sounds)]
        y_vals = []
        for step, (x, y) in enumerate(data_loader):
            num_frames = x.shape[1]
            x = x.to(device).to(torch.float32)
            y = y.to(device).to(torch.float32).unsqueeze(0)
            y_vals.append(int(y.item()))
            # y_vals.extend(y.cpu().numpy().tolist())  # Collect all y values for accuracy calculation

            for i, sound_type in enumerate(sound_types):
                if sound_type == "silence":
                    mel = generate_mel_for_frames(num_frames, silence=True)
                else:
                    mel = generate_mel_for_frames(num_frames, silence=False)
                mel = mel.to(device).to(torch.float32).unsqueeze(0)
                a, v = model(mel, x)
                loss = gru2.cosine_loss(a, v, y)
                # losses[i].append(loss.item())
                losses[i].append(loss.cpu().item())  # Store loss on CPU to avoid GPU memory issues
            # if step == 5:
            #     break
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

def best_accuracy(losses, true_y, flip=False, thresholds=np.arange(0.0, 1.2, 0.1)):
    best_acc = 0
    best_threshold = 0
    for threshold in thresholds:
        if flip:
            results = [1.0 if loss < threshold else 0.0 for loss in losses]
        else:
            results = [0.0 if loss < threshold else 1.0 for loss in losses]
        acc = accuracy(true_y, results)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    print(f"Best accuracy: {best_acc} at threshold: {best_threshold}")
    return best_acc, best_threshold

def test_accuracy(losses, true_y, threshold, flip=False):
    if flip:
        results = [1.0 if loss < threshold else 0.0 for loss in losses]
    else:
        results = [0.0 if loss < threshold else 1.0 for loss in losses]
    acc = accuracy(true_y, results)
    return acc

def all_accuracies(losses, ys):
    for i, sound_type in enumerate(sound_types):
        print(f"Sound type: {sound_type}")
        # print("Losses:", losses[i])
        # print("True labels:", ys)
        best_accuracy(losses[i], ys, flip=False)
        # best_accuracy(losses[i], ys, flip=True) # Seems to be the wrong driection, left in case we make adjustments later

def train_threshold(losses_train, true_y_train, losses_test, true_y_test, thresholds=np.arange(0.0, 1.2, 0.1)):
    best_acc_train, best_threshold = best_accuracy(losses_train, true_y_train, flip=False, thresholds=thresholds)
    acc_test = test_accuracy(losses_test, true_y_test, best_threshold, flip=False)
    print(f"Train accuracy: {best_acc_train}, Test accuracy: {acc_test} at threshold: {best_threshold}")
    return best_threshold, acc_test

def train_threshold_all_sound_types(losses_train, true_y_train, losses_test, true_y_test, thresholds=np.arange(0.0, 1.2, 0.1)):
    best_thresholds = []
    acc_tests = []
    for i in range(len(sound_types)):
        print(f"Sound type: {sound_types[i]}")
        best_threshold, acc_test = train_threshold(losses_train[i], true_y_train, losses_test[i], true_y_test, thresholds)
        best_thresholds.append(best_threshold)
        acc_tests.append(acc_test)
    return best_thresholds, acc_tests

if __name__ == "__main__":
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = hparams.batch_size

    if checkpoint_path  is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                        lr=hparams.syncnet_lr)

    print("Loading checkpoint path")
    if checkpoint_path  is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
    gru2.load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)
    print("Loaded checkpoint from: {}".format(checkpoint_path))
    model.eval()
    

    test_dataset = Dataset_Full_Video(data_root, ground_truth) # Needs a batch size of 1
    train_dataset = Dataset_Full_Video(train_data_root, ground_truth_train)
    print(f"Number of samples in full video dataset: {len(test_dataset)}")
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=1,
        num_workers=8, shuffle=True)
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=1,
        num_workers=8, shuffle=True)

    print("Finding threshold on test data (cheating for comparison purposes) for full video dataset")
    losses, y_vals = audio_loop(model, test_data_loader)
    # print(losses)
    # print(y_vals)
    print("Test data accuracies:")
    all_accuracies(losses, y_vals)
    print()
    print("Training data accuracies:")
    train_losses, train_y_vals = audio_loop(model, train_data_loader)
    all_accuracies(losses, y_vals)

    print("\nTest data accuracies with training data threshold:")
    train_threshold_all_sound_types(train_losses, train_y_vals, losses, y_vals)

    print()
    print()

    test_dataset_5_frame = Dataset_5_Frame(data_root, ground_truth)
    batch_size = 1
    print(f"Number of samples in 5-frame dataset: {len(test_dataset_5_frame)}")
    test_data_loader_5_frame = data_utils.DataLoader(
        test_dataset_5_frame, batch_size=batch_size,
        num_workers=8, drop_last=True, shuffle=True)
    train_dataset_5_frame = Dataset_5_Frame(train_data_root, ground_truth_train)
    train_data_loader_5_frame = data_utils.DataLoader(
        train_dataset_5_frame, batch_size=batch_size,
        num_workers=8, drop_last=True, shuffle=True)

    print("Finding threshold on test data (cheating for comparison purposes) for 5-frame dataset")
    losses_5_frame, y_vals_5_frame = audio_loop(model, test_data_loader_5_frame)
    print("Test data accuracies:")
    all_accuracies(losses_5_frame, y_vals_5_frame)
    print()
    print("Training data accuracies:")
    train_losses_5_frame, train_y_vals_5_frame = audio_loop(model, train_data_loader_5_frame)
    all_accuracies(losses_5_frame, y_vals_5_frame)

    print("\nTest data accuracies with training data threshold:")
    train_threshold_all_sound_types(train_losses_5_frame, train_y_vals_5_frame, losses_5_frame, y_vals_5_frame)