from models import SyncNet_landmarks_gru2 as SyncNet
from lmks_audio_eval import accuracy, best_accuracy, match_digits, get_files_list
import landmarks_audio as audio
from hparams import hparams
from models import SyncNet_landmarks_gru2 as SyncNet
import landmarks_syncnet_train_gru2 as gru2

import torch
import torch.utils.data as data_utils
from torch import optim
import os
import numpy as np
import random

data_root = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_vvadlrs3/main/x_test/'
ground_truth = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_vvadlrs3/main/y_test.npy'
checkpoint_dir = 'landmarks_checkpoints_gru2'
checkpoint_path = None

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
        


def generate_mel_for_frames(num_frames, silence = True, video_fps=hparams.fps, mel_fps=80, sample_rate=16000, hop_length=200):
    """
    Generate a mel spectrogram for silent audio matching `num_frames` video frames.

    Args:
        num_frames (int): Number of video frames.
        video_fps (int): Video frame rate. Default 25.
        mel_fps (int): Mel spectrogram frame rate. Default 80.
        sample_rate (int): Audio sample rate. Default 16000.
        hop_length (int): Hop length used in mel spectrogram. Default 200 (16kHz / 80).

    Returns:
        torch.Tensor: Mel spectrogram of shape [1, 80, time_steps] where time_steps matches the video clip.
    """
    # Compute how many mel frames are needed
    mel_frames = int(num_frames * mel_fps / video_fps)

    # Compute how many audio samples are needed to get those mel frames
    num_samples = (mel_frames - 1) * hop_length  # +1 mel frame per hop

    # Generate silent audio
    if silence:
        gen_audio = torch.zeros(num_samples)
    else:
        gen_audio = torch.randn(num_samples) # Generate white noise

    # Compute mel spectrogram
    mel = audio.melspectrogram(gen_audio).T  # [Time, Mel]
    mel = mel[:mel_frames]  # Clip to exact mel_frames
    mel = torch.FloatTensor(mel.T).unsqueeze(0)  # [1, 80, mel_frames]
    return mel


# x shape: torch.Size([1, 5, 187]), mel shape: torch.Size([1, 1, 80, 16])

if __name__ == "__main__":
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    test_dataset = Dataset(data_root, ground_truth)
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=1,
        num_workers=8)
    
    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                        lr=hparams.syncnet_lr)

    print("Loading checkpoint path")
    if checkpoint_path  is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    if True: #device == 'cuda':
        gru2.load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)
        print("Loaded checkpoint from: {}".format(checkpoint_path))
    

    model.eval()
    
    silent_losses = []
    white_noise_losses = []
    ys = []
    for step, (x, y) in enumerate(test_data_loader):
        num_frames = x.shape[1]
        silent_mel = generate_mel_for_frames(num_frames, silence=True).to(device).unsqueeze(0)  # Add batch dimension
        white_noise_mel = generate_mel_for_frames(num_frames, silence=False).to(device).unsqueeze(0)  # Add batch dimension

        x = x.to(device).to(torch.float32) # Ensure x is float32 and add batch dimension
        y = y.to(device).to(torch.float32).unsqueeze(0)  # Ensure y is float32
        ys.append(y.item())

        a, v = model(silent_mel, x)
        loss = gru2.cosine_loss(a, v, y)
        silent_loss = loss.item()
        silent_losses.append(silent_loss)

        a, v = model(white_noise_mel, x)
        loss = gru2.cosine_loss(a, v, y)
        white_noise_loss = loss.item()
        white_noise_losses.append(white_noise_loss)

print("Silent losses:")
best_accuracy(silent_losses, ys)
best_accuracy(silent_losses, ys, flip=True)
print("White noise losses:")
best_accuracy(white_noise_losses, ys)
best_accuracy(white_noise_losses, ys, flip=True)
# Check the accuracy function for landmarks_syncent_eval