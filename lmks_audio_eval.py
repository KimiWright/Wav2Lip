import landmarks_audio as audio
import torch
import os, argparse
from torch import optim
from hparams import hparams
import landmarks_syncnet_train_gru2 as gru2
from models import SyncNet_landmarks_gru2 as SyncNet
import torch.utils.data as data_utils
import numpy as np
import random
import re


parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')
parser.add_argument("--data_root", help="Root folder of the preprocessed landmarks for LRS3 VVAD dataset", default='/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_vvadlrs3/main/x_test/')
parser.add_argument('--ground_truth', help="Ground truth folder of the LRS3 VVAD dataset", default='/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_vvadlrs3/main/y_test.npy')
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default='landmarks_checkpoints_gru2', type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)

args = parser.parse_args()

checkpoint_dir = args.checkpoint_dir
checkpoint_path = args.checkpoint_path

syncnet_mel_step_size = 16
batch_size = 1#hparams.syncnet_batch_size

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

# No still face or audio dataset
class Moving_Face_Dataset(object):
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

            y = self.y[idx]

            return x_video, y

def crop_audio_window(spec, start_frame_num):
        
    start_idx = int(80. * (start_frame_num / float(hparams.fps)))

    end_idx = start_idx + syncnet_mel_step_size

    return spec[start_idx : end_idx, :]

def cropped_mel(audio_tensor, start_frame_num=0):
    mel = audio.melspectrogram(audio_tensor).T # shape: (Time, Mel)
    cropped_mel = crop_audio_window(mel.copy(), start_frame_num)
    mel = torch.FloatTensor(cropped_mel.T).unsqueeze(0)  # [1, Mel, Time]
    return mel

def accuracy(ys, test_ys):
    correct = 0
    for y, test_y in zip(ys, test_ys):
        if y == test_y:
            correct += 1
    return correct / len(ys)

def true_positive_rate(ys, test_ys):
    true_positive = 0
    for y, test_y in zip(ys, test_ys):
        if y == 1 and test_y == 1:
            true_positive += 1
    return true_positive / len(ys)

def true_negative_rate(ys, test_ys):
    true_negative = 0
    for y, test_y in zip(ys, test_ys):
        if y == 0 and test_y == 0:
            true_negative += 1
    return true_negative / len(ys)

def best_accuracy(silent_losses, white_noise_losses, ys, thresholds):
    best_threshold_silent = 0.5
    best_accuracy_silent = -1.0
    best_threshold_white_noise = 0.5
    best_accuracy_white_noise = -1.0
    best_true_negative_rate_white_noise = -1.0
    best_true_negative_rate_silent = -1.0
    best_true_positive_rate_silent = -1.0
    best_true_positive_rate_white_noise = -1.0

    print("Starting evaluation")
    for threshold in thresholds:
        silent_ys = []
        white_noise_ys = []

        for silent_loss, white_noise_loss in zip(silent_losses, white_noise_losses):
            silent_y = int(silent_loss < threshold)
            white_noise_y = int(white_noise_loss < threshold)
            silent_ys.append(silent_y)
            white_noise_ys.append(white_noise_y)

        silent_accuracy = accuracy(silent_ys, test_ys=ys)
        white_noise_accuracy = accuracy(white_noise_ys, test_ys=ys)

        if silent_accuracy > best_accuracy_silent:
            best_accuracy_silent = silent_accuracy
            best_threshold_silent = threshold

        if white_noise_accuracy > best_accuracy_white_noise:
            best_accuracy_white_noise = white_noise_accuracy
            best_threshold_white_noise = threshold

        true_positive_rate_silent = true_positive_rate(silent_ys, test_ys=ys)
        true_positive_rate_white_noise = true_positive_rate(white_noise_ys, test_ys=ys)
        true_negative_rate_silent = true_negative_rate(silent_ys, test_ys=ys)
        true_negative_rate_white_noise = true_negative_rate(white_noise_ys, test_ys=ys)

        best_true_positive_rate_silent = max(best_true_positive_rate_silent, true_positive_rate_silent)
        best_true_positive_rate_white_noise = max(best_true_positive_rate_white_noise, true_positive_rate_white_noise)
        best_true_negative_rate_silent = max(best_true_negative_rate_silent, true_negative_rate_silent)
        best_true_negative_rate_white_noise = max(best_true_negative_rate_white_noise, true_negative_rate_white_noise)

    print("Final silent accuracy: {}, Final white noise accuracy: {}".format(best_accuracy_silent, best_accuracy_white_noise))
    print("Best silent threshold: {}, Best white noise threshold: {}".format(best_threshold_silent, best_threshold_white_noise))
    print("Best silent true negative rate: {}, Best white noise true negative rate: {}".format(
        best_true_negative_rate_silent, best_true_negative_rate_white_noise))


def best_accuracy_less_than(silent_losses, white_noise_losses, ys, thresholds):
    silent_ys = []
    white_noise_ys = []
    thresholds = np.arange(0.0, 1.0, 0.01)
    best_thereshold_silent = 0.5
    best_accuracy_silent = -1.0
    best_thereshold_white_noise = 0.5
    best_accuracy_white_noise = -1.
    best_true_negative_rate_white_noise = -1.
    best_true_negative_rate_silent = -1.
    best_true_positive_rate_silent = -1.
    best_true_positive_rate_white_noise = -1.
    print("Starting Less Than evaluation")
    for threshold in thresholds:
        for silent_loss, white_noise_loss in zip(silent_losses, white_noise_losses):
            silent_y = int(silent_loss > threshold)
            white_noise_y = int(white_noise_loss > threshold)
            silent_ys.append(silent_y)
            white_noise_ys.append(white_noise_y)
        silent_accuracy = accuracy(silent_ys, test_ys=ys)
        white_noise_accuracy = accuracy(white_noise_ys, test_ys=ys)
        if silent_accuracy > best_accuracy_silent:
            best_accuracy_silent = silent_accuracy
            best_thereshold_silent = threshold
        if white_noise_accuracy > best_accuracy_white_noise:
            best_accuracy_white_noise = white_noise_accuracy
            best_thereshold_white_noise = threshold
        true_positive_rate_silent = true_positive_rate(silent_ys, test_ys=ys)
        true_positive_rate_white_noise = true_positive_rate(white_noise_ys, test_ys=ys)
        true_negative_rate_silent = true_negative_rate(silent_ys, test_ys=ys)
        true_negative_rate_white_noise = true_negative_rate(white_noise_ys, test_ys=ys)
        if true_positive_rate_silent > best_true_positive_rate_silent:
            best_true_positive_rate_silent = true_positive_rate_silent
        if true_positive_rate_white_noise > best_true_positive_rate_white_noise:
            best_true_positive_rate_white_noise = true_positive_rate_white_noise
        if true_negative_rate_silent > best_true_negative_rate_silent:
            best_true_negative_rate_silent = true_negative_rate_silent
        if true_negative_rate_white_noise > best_true_negative_rate_white_noise:
            best_true_negative_rate_white_noise = true_negative_rate_white_noise
        # print("Threshold: {}".format(threshold))
        # print("Silent accuracy: {}, White noise accuracy: {}".format(silent_accuracy, white_noise_accuracy))
    print("Final silent accuracy: {}, Final white noise accuracy: {}".format(best_accuracy_silent, best_accuracy_white_noise))
    print("Best silent threshold: {}, Best white noise threshold: {}".format(best_thereshold_silent, best_thereshold_white_noise))
    print("Best silent true negative rate: {}, Best white noise true negative rate: {}".format(best_true_negative_rate_silent, best_true_negative_rate_white_noise))

if __name__ == "__main__":
    # Generate 1 second of silence
    silence = torch.zeros(16000)  # 1 second at 16kHz
    white_noise = torch.randn(16000)

    ### Make a 5 frame Mel spectrogram and a full one for comparison ###
    # Starting with the 5 frame.

    test_dataset = gru2.Dataset('val')

    device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=8)
    model.eval()

    silent_mel = cropped_mel(silence, start_frame_num=0).to(device) # shape: (1, Mel, Time)
    silent_mel = silent_mel.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, Mel, Time]
    white_noise_mel = cropped_mel(white_noise, start_frame_num=0).to(device) # shape: (1, Mel, Time)
    white_noise_mel = white_noise_mel.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, 1, Mel, Time]

    ys = []
    silent_losses = []
    white_noise_losses = []
    for step, (x, mel, y) in enumerate(test_data_loader):        
        x = x.to(device)
        mel = mel.to(device)


        a, v = model(mel, x)
        y = y.to(device)
        loss = gru2.cosine_loss(a, v, y)

        silent_a, silent_v = model(silent_mel, x)
        white_noise_a, white_noise_v = model(white_noise_mel, x)

        silent_loss = gru2.cosine_loss(silent_a, silent_v, y)
        white_noise_loss = gru2.cosine_loss(white_noise_a, white_noise_v, y)
        silent_losses.append(silent_loss.item())
        white_noise_losses.append(white_noise_loss.item())
        y = int(y.item())

        ys.append(y)

    print()
    print("Regular dataset")
    best_accuracy(silent_losses, white_noise_losses, ys, thresholds=np.arange(0.0, 1.0, 0.01))
    best_accuracy_less_than(silent_losses, white_noise_losses, ys, thresholds=np.arange(0.0, 1.0, 0.01))

    mf_test_dataset = Moving_Face_Dataset(args.data_root, args.ground_truth)
    mf_test_data_loader = data_utils.DataLoader(
        mf_test_dataset, batch_size=batch_size,
        num_workers=8)
    print("Moving face dataset loaded")

    silent_losses = []
    white_noise_losses = []
    ys = []
    for step, (x, y) in enumerate(mf_test_data_loader):
        x = x.to(device).to(torch.float32)
        y = y.to(device).to(torch.float32)
        # print(x.dtype, y.dtype)
        # print(silent_mel.dtype, white_noise_mel.dtype)
        silent_a, silent_v = model(silent_mel, x)
        white_noise_a, white_noise_v = model(white_noise_mel, x)
        if y.shape == torch.Size([1]):
            y = y.unsqueeze(0)

        silent_loss = gru2.cosine_loss(silent_a, silent_v, y)
        white_noise_loss = gru2.cosine_loss(white_noise_a, white_noise_v, y)
        silent_losses.append(silent_loss.item())
        white_noise_losses.append(white_noise_loss.item())
        y = int(y.item())

        ys.append(y)

    print()
    print("Moving face dataset")
    best_accuracy(silent_losses, white_noise_losses, ys, thresholds=np.arange(0.0, 1.0, 0.01))
    best_accuracy_less_than(silent_losses, white_noise_losses, ys, thresholds=np.arange(0.0, 1.0, 0.01))



## Make another one using the whole video
## Why are we using syncnet instead of a model that is trained on VVAD?
    
    