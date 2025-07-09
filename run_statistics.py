import os
import re
import numpy as np
from tqdm import tqdm

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

        print(idx, end='\r')  # Print the index to track progress
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

        data.append((x_video, y[idx]))
        if data_point_limit is not None and len(data) >= data_point_limit:
            break

    return data

################
# Datasets
#################

data_out_test = get_data(data_root, ground_truth)
data_out_train = get_data(train_data_root, ground_truth_train, data_point_limit=None, start_idx=2000)

class Dataset_Full_Video(object):
    def __init__(self, split = 'test'):
        if split == 'test':
            self.data = data_out_test
        elif split == 'train':
            self.data = data_out_train
        else:
            raise ValueError("Split must be 'test' or 'train'")

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x_video, y = self.data[idx]
        return x_video, y
    
class Dataset_5_Frame(object):
    def __init__(self, split = 'test'):
        if split == 'test':
            self.data = data_out_test
        elif split == 'train':
            self.data = data_out_train
        else:
            raise ValueError("Split must be 'test' or 'train'")

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # print(f"Getting item {idx} from dataset")
        x_video, y = self.data[idx]
        if x_video is None:
            raise ValueError(f"x_video is None for index {idx}")
        # print(f"x_video shape before windowing: {x_video.shape}")
        x_video = get_window_npy(x_video, start_id=0)
        if x_video is None:
            raise ValueError(f"x_video is None after windowing for index {idx}")
        # print(f"x_video shape: {x_video.shape}")
        return x_video, y
    
class Dataset_5_Frame_Chunks(object):
    def __init__(self, split = 'test'):
        if split == 'test':
            self.data = data_out_test
        elif split == 'train':
            self.data = data_out_train
        else:
            raise ValueError("Split must be 'test' or 'train'")

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x_video, y = self.data[idx]
        x_video_chunks = []
        num_frames = x_video.shape[0]
        for start_id in range(0, num_frames, 5):  # Get chunks of 5 frames
            chunk = get_window_npy(x_video, start_id=start_id)
            if chunk is not None:
                x_video_chunks.append(chunk)
        x_video_chunks = np.array(x_video_chunks)
        return x_video_chunks, y

#################
# Audio generation functions
#################

def crop_audio_window(spec, num_frames=5, start_frame_num=0, video_fps=hparams.fps, mel_fps=80):
        mel_frames = int(num_frames * mel_fps / video_fps)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        end_idx = start_idx + mel_frames
        return spec[start_idx : end_idx, :]

babble_noise = '/home/ksw38/groups/grp_lip/nobackup/archive/datasets/speech-commands/_background_noise_/babble_noise.wav'
babble_wave = audio.load_wav(babble_noise, hparams.sample_rate)
babble_mel_global = audio.melspectrogram(babble_wave).T  # [Time, Mel]
def generate_babble_mel(num_frames=5, start_frame_num=0, video_fps=hparams.fps, mel_fps=80):
    babble_mel = crop_audio_window(babble_mel_global.copy(), num_frames=num_frames, start_frame_num=start_frame_num, video_fps=video_fps, mel_fps=mel_fps)  # Crop to the first mel step
    babble_mel = torch.FloatTensor(babble_mel.T).unsqueeze(0)  # [1, Mel, Time]
    return babble_mel

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

sound_types = ["silence", "white_noise", "babble_noise"]

###################
# Audio loop for evaluation
###################

def audio_loop(model, data_loader, device): # Try the loss from contrastive learning
    with torch.no_grad():
        num_sounds = len(sound_types)
        losses = [[] for _ in range(num_sounds)]
        y_vals = []
        for step, (x, y) in enumerate(data_loader):
            num_frames = x.shape[1]
            x = x.to(device).to(torch.float32)
            y = y.to(device).to(torch.float32).unsqueeze(0)
            y_vals.append(int(y.item()))

            for i, sound_type in enumerate(sound_types):
                if sound_type == "silence":
                    mel = generate_mel_for_frames(num_frames, silence=True)
                elif sound_type == "babble_noise":
                    mel = generate_babble_mel(num_frames, start_frame_num=0)
                else:
                    mel = generate_mel_for_frames(num_frames, silence=False)
                mel = mel.to(device).to(torch.float32).unsqueeze(0)
                a, v = model(mel, x)
                loss = gru2.cosine_loss(a, v, y)
                losses[i].append(loss.cpu().item())  # Store loss on CPU to avoid GPU memory issues
        return losses, y_vals
    
def audio_loop_5_frame_chunk(model, data_loader, device): # Try the loss from contrastive learning
    with torch.no_grad():
        num_sounds = len(sound_types)
        losses = [[] for _ in range(num_sounds)]
        av_val_lists = [[] for _ in range(num_sounds)]
        y_vals = []
        for step, (x, y) in enumerate(data_loader):
            num_frames = 5#x.shape[1]
            x = x.to(device).to(torch.float32)
            print(x.shape) # [Batch, Chunk, Frames, Features]
            # Change to [Chunk, Batch, Frames, Features] for easier iteration
            x = x.permute(1, 0, 2, 3)  # [Chunk, Batch, Frames, Features]
            y = y.to(device).to(torch.float32).unsqueeze(0)
            y_vals.append(int(y.item()))

            for i, sound_type in enumerate(sound_types):
                if sound_type == "silence":
                    mel = generate_mel_for_frames(num_frames, silence=True)
                elif sound_type == "babble_noise":
                    mel = generate_babble_mel(num_frames, start_frame_num=0)
                else:
                    mel = generate_mel_for_frames(num_frames, silence=False)
                mel = mel.to(device).to(torch.float32).unsqueeze(0)

                av_vals = []
                for j in range(x.shape[0]):  # Iterate over the chunks
                    try:
                        a, v = model(mel, x[j])  # x[j] is now [Batch, Frames, Features]
                        av_vals.append((a, v))
                    except Exception as e:
                        print(f"Error processing chunk {j} in step {step}: {e}")
                        continue
                av_val_lists[i].append(av_vals)
        return y_vals, av_val_lists
    
def chunk_losses(y_vals, av_val_list):
    losses = [[] for _ in range(len(sound_types))]
    for i, av_vals in enumerate(av_val_list): # Iterate over sound types
            # print(f"{sound_types[i]} Number of videos: {len(av_vals)}")
            # print(len(y_vals))
            for j, video in enumerate(av_vals):  # Iterate over videos
                # for chunk in video:
                #     print(chunk[0].shape, chunk[1].shape) # a and v shapes
                a_vals, v_vals = zip(*video)  # Unzip a and v values
                a_vals = torch.stack(a_vals, dim=0)  # [Chunks, Batch, 128]
                v_vals = torch.stack(v_vals, dim=0)  # [Chunks, Batch, 128]
                a_mean = a_vals.mean(dim=0)  # Average over Chunks
                v_mean = v_vals.mean(dim=0)  # Average over Batch
                y = torch.Tensor([y_vals[j]]).unsqueeze(0)
                a_mean = a_mean.to(device)
                v_mean = v_mean.to(device)
                y = y.to(device)
                loss = gru2.cosine_loss(a_mean, v_mean, y)
                # print(f"Loss for video {j}: {loss.item()}")
                losses[i].append(loss.cpu().item())  # Store loss on CPU to avoid GPU memory issues
    return losses

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
    # print(f"Thresholds: {thresholds}")
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

print() #Provide visual separation from the prepatory code

if __name__ == "__main__":
    device = 'cpu'# torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shuffle_dataset = False
    num_workers = 1

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
    

    test_dataset = Dataset_Full_Video('test')
    train_dataset = Dataset_Full_Video('train') # Causes problemes in audio loop
    print(f"Number of samples in full video dataset: {len(test_dataset)}")
    print(f"Number of samples in training dataset: {len(train_dataset)}")
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=1,
        num_workers=num_workers, shuffle=shuffle_dataset)
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=1,
        num_workers=num_workers, shuffle=shuffle_dataset)

    ## Configure which tests to perform
    test_accuracy_threshold_on_test_data = True
    train_accuracy_threshold_on_train_data = True
    test_accuracy_threshold_on_train_data = True
    find_test_losses = test_accuracy_threshold_on_test_data or test_accuracy_threshold_on_train_data
    find_train_losses = train_accuracy_threshold_on_train_data or test_accuracy_threshold_on_train_data

    # print("Finding losses for full video dataset")
    # if find_test_losses:
    #     losses, y_vals = audio_loop(model, test_data_loader, device)
    # if find_train_losses:
    #     train_losses, train_y_vals = audio_loop(model, train_data_loader, device)

    # if test_accuracy_threshold_on_test_data:
    #     print("Finding threshold on test data (cheating for comparison purposes) for full video dataset")
    #     print("Test data accuracies:")
    #     all_accuracies(losses, y_vals)
    #     print()
    # if train_accuracy_threshold_on_train_data:
    #     print("Training data accuracies:")
    #     all_accuracies(losses, y_vals)

    # if test_accuracy_threshold_on_train_data:
    #     min_threshold = min(np.min(losses[i]) for i in range(len(sound_types)))
    #     max_threshold = max(np.max(losses[i]) for i in range(len(sound_types)))
    #     threshold_range = np.arange(min_threshold, max_threshold + 0.1, 0.1)
    #     print("\nTest data accuracies with training data threshold:")
    #     train_threshold_all_sound_types(train_losses, train_y_vals, losses, y_vals, thresholds=threshold_range)
    #     print()
    # print()

    test_dataset_5_frame = Dataset_5_Frame('test')
    batch_size = 1
    print(f"Number of samples in 5-frame dataset: {len(test_dataset_5_frame)}")
    print(f"Number of samples in training 5-frame dataset: {len(train_dataset)}")
    test_data_loader_5_frame = data_utils.DataLoader(
        test_dataset_5_frame, batch_size=batch_size,
        num_workers=8, drop_last=True, shuffle=shuffle_dataset)
    train_dataset_5_frame = Dataset_5_Frame('train')
    train_data_loader_5_frame = data_utils.DataLoader(
        train_dataset_5_frame, batch_size=batch_size,
        num_workers=8, drop_last=True, shuffle=shuffle_dataset)
    
    print("Begin Data Testing Loop (Find the samples that cause problems)")
    for i, (x, y) in enumerate(train_data_loader_5_frame):
        # if i < 2245:
        #     continue
        x = x.to(device).to(torch.float32)
        num_frames = x.shape[1]
        mel = generate_mel_for_frames(num_frames, silence=True)
        mel = mel.to(device).to(torch.float32).unsqueeze(0)
        print(f"Step {i}, x shape: {x.shape}, mel shape: {mel.shape}")
        # a, v = model(mel, x)
    
    print("Finding losses for 5-frame dataset")
    if find_test_losses:
        losses_5_frame, y_vals_5_frame = audio_loop(model, test_data_loader_5_frame, device)
    if find_train_losses:
        train_losses_5_frame, train_y_vals_5_frame = audio_loop(model, train_data_loader_5_frame, device)

    if test_accuracy_threshold_on_test_data:
        print("Finding threshold on test data (cheating for comparison purposes) for 5-frame dataset")
        print("Test data accuracies:")
        all_accuracies(losses_5_frame, y_vals_5_frame)
        print()
    if train_accuracy_threshold_on_train_data:
        print("Training data accuracies:")
        all_accuracies(train_losses_5_frame, train_y_vals_5_frame)

    if test_accuracy_threshold_on_train_data:
        min_threshold = min(np.min(losses[i]) for i in range(len(sound_types)))
        max_threshold = max(np.max(losses[i]) for i in range(len(sound_types)))
        threshold_range = np.arange(min_threshold, max_threshold + 0.1, 0.1)
        print("\nTest data accuracies with training data threshold:")
        train_threshold_all_sound_types(train_losses_5_frame, train_y_vals_5_frame, losses_5_frame, y_vals_5_frame, thresholds=threshold_range)

    # test_dataset_5_frame_chunks = Dataset_5_Frame_Chunks('test')
    # print(f"\nNumber of samples in 5-frame chunks dataset: {len(test_dataset_5_frame_chunks)}")
    # train_dataset_5_frame_chunks = Dataset_5_Frame_Chunks('train')
    # print(f"Number of samples in training 5-frame chunks dataset: {len(train_dataset_5_frame_chunks)}")
    
    # test_data_loader_5_frame_chunks = data_utils.DataLoader(
    #     test_dataset_5_frame_chunks, batch_size=batch_size,
    #     num_workers=8, drop_last=True, shuffle=shuffle_dataset)
    # train_data_loader_5_frame_chunks = data_utils.DataLoader(
    #     train_dataset_5_frame_chunks, batch_size=batch_size,
    #     num_workers=8, drop_last=True, shuffle=shuffle_dataset)
    

    # print("\nFinding losses for 5-frame chunks dataset")
    # test_accuracy_threshold_on_train_data = True

    # find_test_losses = True
    # find_train_losses = True
    # if find_test_losses:
    #     y_vals_5_frame_chunks, av_val_list = audio_loop_5_frame_chunk(model, test_data_loader_5_frame_chunks, device)
    #     print("Test losses found for 5-frame chunks dataset")
    #     losses_5_frame_chunks = chunk_losses(y_vals_5_frame_chunks, av_val_list)

    # if find_train_losses:
    #     train_y_vals_5_frame_chunks, av_val_list = audio_loop_5_frame_chunk(model, train_data_loader_5_frame_chunks, device)
    #     print("Train losses found for 5-frame chunks dataset")
    #     train_losses_5_frame_chunks = chunk_losses(train_y_vals_5_frame_chunks, av_val_list)
        

    # if test_accuracy_threshold_on_train_data:
    #     min_threshold = min(np.min(losses_5_frame_chunks[i]) for i in range(len(sound_types)))
    #     max_threshold = max(np.max(losses_5_frame_chunks[i]) for i in range(len(sound_types)))
    #     threshold_range = np.arange(min_threshold, max_threshold + 0.1, 0.1)
    #     print("\nTest data accuracies with training data threshold:")
    #     train_threshold_all_sound_types(train_losses_5_frame_chunks, train_y_vals_5_frame_chunks, losses_5_frame_chunks, y_vals_5_frame_chunks, thresholds=threshold_range)
