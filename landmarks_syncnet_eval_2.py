# This file evaluates using a still frame against a video sequence for VVAD LRS3 dataset.
# Designed as a complement to run_statistics.py which uses audio for evaluation.
import numpy as np
import os
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

import run_statistics as run_stats
from models import lmks_only as lmks_only
from hparams import hparams


data_out_test = run_stats.data_out_test
data_out_train = run_stats.data_out_train
checkpoint_path = run_stats.checkpoint_path
checkpoint_dir = run_stats.checkpoint_dir
data_point_limit = None # Limit the number of data points to process for testing

#############
# Loading functions
#############
def load_face_model(checkpoint_path, device, startswith='face'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    full_state_dict = checkpoint['state_dict']

    face_state_dict = {k: v for k, v in full_state_dict.items() if k.startswith("face")}

    model = lmks_only().to(device)
    missing, unexpected = model.load_state_dict(face_state_dict, strict=False)

    if missing:
        print("Trying to load with new keys...")
        new_state_dict = {}
        for k, v in face_state_dict.items():
            new_key = k.split('.', 1)[1] if '.' in k else k
            new_state_dict[new_key] = v
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    if missing:
        print("Missing keys in the state_dict:", missing)
    if unexpected:
        print("Unexpected keys in the state_dict:", unexpected)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    return model

#############
# Dataset
#############
class Dataset_Still_Face(object):
    def __init__(self, split = 'test'):
        global data_out_test, data_out_train, data_point_limit
        if split == 'test':
            self.data = data_out_test
            if len(self.data) == 0:
                data_out_test = run_stats.get_data(run_stats.data_root, run_stats.ground_truth, data_point_limit=data_point_limit)
                self.data = data_out_test
        elif split == 'train':
            self.data = data_out_train
            if len(self.data) == 0:
                data_out_train = run_stats.get_data(run_stats.train_data_root, run_stats.ground_truth_train, data_point_limit=data_point_limit)
                self.data = data_out_train
        else:
            raise ValueError("Split must be 'test' or 'train'")
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x_video, y = self.data[idx]
        num_frames = x_video.shape[0]
        x_still = np.tile(x_video[0], (num_frames, 1)) # Make a still video of the first frame
        return x_video, x_still, y
    
class Dataset_Still_Face_5_Frames(object):
    def __init__(self, split = 'test'):
        global data_out_test, data_out_train, data_point_limit
        if split == 'test':
            self.data = data_out_test
            if len(self.data) == 0:
                data_out_test = run_stats.get_data(run_stats.data_root, run_stats.ground_truth, data_point_limit=data_point_limit)
                self.data = data_out_test
        elif split == 'train':
            self.data = data_out_train
            if len(self.data) == 0:
                data_out_train = run_stats.get_data(run_stats.train_data_root, run_stats.ground_truth_train, data_point_limit=data_point_limit)
                self.data = data_out_train
        else:
            raise ValueError("Split must be 'test' or 'train'")
        
        self.processed_data = []
        for datum in self.data:
            x_video_full, y = datum
            if x_video_full is None:
                raise ValueError("x_video_full is None")
            x_video = run_stats.get_window_npy(x_video_full, start_id=0)
            if x_video is not None:
                self.processed_data.append((x_video, y))
            # else:
                        #     print(f"Skipping data point {x_video_full.shape} due to insufficient frames")
        
    def __len__(self):
        return len(self.processed_data)
    def __getitem__(self, idx):
        x_video, y = self.processed_data[idx]
        num_frames = x_video.shape[0]
        x_still = np.tile(x_video[0], (num_frames, 1)) # Make a still video of the first frame
        return x_video, x_still, y
    
class Dataset_Still_Face_5_Frame_Chunks(object):
    def __init__(self, split = 'test'):
        global data_out_test, data_out_train, data_point_limit
        if split == 'test':
            self.data = data_out_test
            if len(self.data) == 0:
                data_out_test = run_stats.get_data(run_stats.data_root, run_stats.ground_truth, data_point_limit=data_point_limit)
                self.data = data_out_test
        elif split == 'train':
            self.data = data_out_train
            if len(self.data) == 0:
                data_out_train = run_stats.get_data(run_stats.train_data_root, run_stats.ground_truth_train, data_point_limit=data_point_limit)
                self.data = data_out_train
        else:
            raise ValueError("Split must be 'test' or 'train'")
        
        self.processed_data = []
        for datum in self.data:
            x_video_full, y = datum
            if x_video_full is None:
                raise ValueError("x_video_full is None")
            x_video_chunks = []
            x_still_chunks = []
            num_frames = x_video_full.shape[0]
            for start_id in range(0, num_frames, 5):  # Get chunks of 5 frames
                chunk = run_stats.get_window_npy(x_video_full, start_id=start_id)
                if chunk is not None:
                    x_video_chunks.append(chunk)
                    x_still_chunks.append(np.tile(chunk[0], (chunk.shape[0], 1)))  # Make a still video of the first frame
            if len(x_video_chunks) > 0:
                self.processed_data.append((np.array(x_video_chunks), np.array(x_still_chunks), y))
            # else:
            #     print(f"Skipping data point {x_video_full.shape} due to insufficient frames for chunks")

    def __len__(self):
        return len(self.processed_data)
    def __getitem__(self, idx):
        x_video, x_still, y = self.processed_data[idx]
        return x_video, x_still, y

##############
# Loss functions
##############

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

###############
# Main evaluation loop
###############
def still_face_evaluation_loop(model, test_data_loader, train_data_loader, device):
    # Main evaluation loop
    with torch.no_grad():
        cosine_losses_train = []
        fconfm_train = []
        y_vals_train = []
        for step, (x_video, x_still, y) in enumerate(train_data_loader):
            x_video = x_video.to(device).to(torch.float32)
            x_still = x_still.to(device).to(torch.float32)
            y = y.to(device)              
            y_vals_train.append(y.item())
            y = y.float().unsqueeze(1) # Ensure y is of shape (batch_size, 1)

            feat_video = model(x_video)
            feat_still = model(x_still)

            fconfm = computeDist(feat_video, feat_still, vshift=15)
            fconfm_train.append(fconfm)
            # loss = cosine_loss(feat_video, feat_still, y)
            loss = F.cosine_similarity(feat_video, feat_still)
            cosine_losses_train.append(loss.item())

        min_fconfm_train = np.min(np.concatenate(fconfm_train))
        max_fconfm_train = np.max(np.concatenate(fconfm_train))
        min_cosine_loss_train = np.min(cosine_losses_train)
        max_cosine_loss_train = np.max(cosine_losses_train)
        fconfm_range = np.arange(min_fconfm_train, max_fconfm_train + 0.1, 0.1)
        cosine_loss_range = np.arange(min_cosine_loss_train, max_cosine_loss_train + 0.1, 0.1)

        print("Training statistics:")
        print("\tFlip is True")
        print("Cosine Loss")
        run_stats.best_accuracy(cosine_losses_train, y_vals_train, flip=True, thresholds=cosine_loss_range)
        print("Fconfm")
        run_stats.best_accuracy(fconfm_train, y_vals_train, flip=True, thresholds=fconfm_range)
        print("\tFlip is False")
        print("Cosine Loss")
        run_stats.best_accuracy(cosine_losses_train, y_vals_train, flip=False, thresholds=cosine_loss_range)
        print("Fconfm")
        run_stats.best_accuracy(fconfm_train, y_vals_train, flip=False, thresholds=fconfm_range)

        fconfm_test = []
        cosine_losses_test = []
        y_vals_test = []
        for step, (x_video, x_still, y) in enumerate(test_data_loader):
            x_video = x_video.to(device).to(torch.float32)
            x_still = x_still.to(device).to(torch.float32)
            y = y.to(device)              
            y_vals_test.append(y.item())
            y = y.float().unsqueeze(1)

            feat_video = model(x_video)
            feat_still = model(x_still)
            fconfm = computeDist(feat_video, feat_still, vshift=15)
            fconfm_test.append(fconfm)
            # loss = cosine_loss(feat_video, feat_still, y)
            loss = F.cosine_similarity(feat_video, feat_still)
            cosine_losses_test.append(loss.item())

        print("Testing statistics:")
        print("\tFlip is True")
        print("Cosine Loss")
        run_stats.train_threshold(cosine_losses_train, y_vals_train, cosine_losses_test, y_vals_test, thresholds=cosine_loss_range, flip=True)
        print("Fconfm")
        run_stats.train_threshold(fconfm_train, y_vals_train, fconfm_test, y_vals_test, thresholds=fconfm_range, flip=True)
        print("\tFlip is False")
        print("Cosine Loss")
        run_stats.train_threshold(cosine_losses_train, y_vals_train, cosine_losses_test, y_vals_test, thresholds=cosine_loss_range, flip=False)
        print("Fconfm")
        run_stats.train_threshold(fconfm_train, y_vals_train, fconfm_test, y_vals_test, thresholds=fconfm_range, flip=False)

def chunk_losses(y_vals, av_vals, device):
    cosine_losses = []
    fconfm_vals = []
    for j, video in enumerate(av_vals):
        a_vals, v_vals = zip(*video)
        a_vals = torch.stack(a_vals, dim=0)  # [Chunks, Batch, 128]
        v_vals = torch.stack(v_vals, dim=0)  # [Chunks, Batch, 128]
        a_mean = a_vals.mean(dim=0)  # Average over Chunks
        v_mean = v_vals.mean(dim=0)  # Average over Batch
        y = torch.Tensor([y_vals[j]]).unsqueeze(0)
        a_mean = a_mean.to(device)
        v_mean = v_mean.to(device)
        y = y.to(device)
        fconfm = computeDist(a_mean, v_mean, vshift=15)
        fconfm_vals.append(fconfm)
        # loss = cosine_loss(a_mean, v_mean, y)
        loss = F.cosine_similarity(a_mean, v_mean)
        cosine_losses.append(loss.item())
    return cosine_losses, fconfm_vals

if __name__ == "__main__":
    # Constants
    device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shuffle_dataset = False
    num_workers = 1

    # Load the model and optimizer and setup data loaders
    if checkpoint_path  is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    model = load_face_model(checkpoint_path, device)
    print(f"Loading model from: {checkpoint_path}")

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)
    model.eval()

    test_dataset = Dataset_Still_Face('test')
    train_dataset = Dataset_Still_Face('train') # Causes problemes in audio loop

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=1,
        num_workers=num_workers, shuffle=shuffle_dataset)
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=1,
        num_workers=num_workers, shuffle=shuffle_dataset)
    
    print("Total training data points:", len(train_dataset))
    print("Total testing data points:", len(test_dataset))
    
    # Run the evaluation loop
    print("\nRunning evaluation loop for Dataset_Still_Face")
    still_face_evaluation_loop(model, test_data_loader, train_data_loader, device)

    test_dataset = Dataset_Still_Face_5_Frames('test')
    train_dataset = Dataset_Still_Face_5_Frames('train')

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=1,
        num_workers=num_workers, shuffle=shuffle_dataset)
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=1,
        num_workers=num_workers, shuffle=shuffle_dataset)
    
    print("Total training data points:", len(train_dataset))
    print("Total testing data points:", len(test_dataset))

    # Run the evaluation loop
    print("\nRunning evaluation loop for Dataset_Still_Face_5_Frames")
    still_face_evaluation_loop(model, test_data_loader, train_data_loader, device)

    test_dataset = Dataset_Still_Face_5_Frame_Chunks('test')
    train_dataset = Dataset_Still_Face_5_Frame_Chunks('train')

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=1,
        num_workers=num_workers, shuffle=shuffle_dataset)
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=1,
        num_workers=num_workers, shuffle=shuffle_dataset)
    
    print("Total training data points:", len(train_dataset))
    print("Total testing data points:", len(test_dataset))

    print("\nRunning evaluation loop for Dataset_Still_Face_5_Frame_Chunks")
    with torch.no_grad():
        y_vals_train = []
        av_val_lists_train = []
        for step, (x_video, x_still, y) in enumerate(train_data_loader):
            x_video = x_video.to(device).to(torch.float32).permute(1, 0, 2, 3) # [Chunk, Batch, Frames, Features]
            x_still = x_still.to(device).to(torch.float32).permute(1, 0, 2, 3) # [Chunk, Batch, Frames, Features]
            y = y.to(device)              
            y = y.float().unsqueeze(1)
            y_vals_train.append(y.item())
            av_vals_train = []
            for j in range(x_video.shape[0]):
                feat_video = model(x_video[j])
                feat_still = model(x_still[j])
                # print(feat_still.shape, feat_video.shape)
                av_vals_train.append((feat_video, feat_still))
            av_val_lists_train.append(av_vals_train)
        cosine_losses, fconfm_vals = chunk_losses(y_vals_train, av_val_lists_train, device)

        min_fconfm_train = np.min(np.concatenate(fconfm_vals))
        max_fconfm_train = np.max(np.concatenate(fconfm_vals))
        min_cosine_loss_train = np.min(cosine_losses)
        max_cosine_loss_train = np.max(cosine_losses)
        fconfm_range = np.arange(min_fconfm_train, max_fconfm_train + 0.1, 0.1)
        cosine_loss_range = np.arange(min_cosine_loss_train, max_cosine_loss_train + 0.1, 0.1) 

        print("Training statistics:")
        print("\tFlip is True")
        print("Cosine Loss")
        run_stats.best_accuracy(cosine_losses, y_vals_train, flip=True, thresholds=cosine_loss_range)
        print("Fconfm")
        run_stats.best_accuracy(fconfm_vals, y_vals_train, flip=True, thresholds=fconfm_range)
        print("\tFlip is False")
        print("Cosine Loss")
        run_stats.best_accuracy(cosine_losses, y_vals_train, flip=False, thresholds=cosine_loss_range)
        print("Fconfm")
        run_stats.best_accuracy(fconfm_vals, y_vals_train, flip=False, thresholds=fconfm_range)

        y_vals_test = []
        av_val_lists_test = []
        for step, (x_video, x_still, y) in enumerate(test_data_loader):
            x_video = x_video.to(device).to(torch.float32).permute(1, 0, 2, 3)
            x_still = x_still.to(device).to(torch.float32).permute(1, 0, 2, 3)
            y = y.to(device)              
            y = y.float().unsqueeze(1)
            y_vals_test.append(y.item())
            av_vals_test = []
            for j in range(x_video.shape[0]):
                feat_video = model(x_video[j])
                feat_still = model(x_still[j])
                av_vals_test.append((feat_video, feat_still))
            av_val_lists_test.append(av_vals_test)
        cosine_losses, fconfm_vals = chunk_losses(y_vals_test, av_val_lists_test, device)  

        print("Testing statistics:")
        print("\tFlip is True")
        print("Cosine Loss")
        run_stats.train_threshold(cosine_losses, y_vals_train, cosine_losses, y_vals_test, thresholds=cosine_loss_range, flip=True)
        print("Fconfm")
        run_stats.train_threshold(fconfm_vals, y_vals_train, fconfm_vals, y_vals_test, thresholds=fconfm_range, flip=True)
        print("\tFlip is False")
        print("Cosine Loss")
        run_stats.train_threshold(cosine_losses, y_vals_train, cosine_losses, y_vals_test, thresholds=cosine_loss_range, flip=False)
        print("Fconfm")
        run_stats.train_threshold(fconfm_vals, y_vals_train, fconfm_vals, y_vals_test, thresholds=fconfm_range, flip=False)