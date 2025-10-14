import os
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION 
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils import data as data_utils
import numpy as np
import csv

from hparams import hparams
import st_gcn_train_mediapipe as st

from models import audio_only
from models import LandmarkSTGCNConformer
from models import build_adjacency

st_gcn_checkpoint = "ckpt_folder/checkpoints_mediapipe"
audio_checkpoint = "ckpt_folder/checkpoints_mediapipe_audio"

test_data_csv = "/home/ksw38/groups/grp_landmarks/nobackup/autodelete/landmarks_vvadlrs3_mp/main/x_test_files.csv"
train_data_csv = "/home/ksw38/groups/grp_landmarks/nobackup/autodelete/landmarks_vvadlrs3_mp/main/x_train_files.csv"

syncnet_T = 5

def get_window_npy(data, syncnet_T, start_id=0):
    if start_id + syncnet_T < len(data):
        return torch.FloatTensor(data[start_id : start_id + syncnet_T])
    else:
        return None

def get_data(csv_path, data_point_limit=None):
    data = []
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file_path = row["file_path"]
            lmks = np.load(file_path)
            y = row["y"]

            data.append((lmks, y))
            if data_point_limit is not None and len(data) >= data_point_limit:
                break
    return data
        
data_out_test = []
data_out_train = []

class Dataset_Frames(object):
    def __init__(self, split = 'test', frames=syncnet_T, data = None, data_point_limit=None):
        if data == None:
            if split == 'test':
                global data_out_test
                self.data = data_out_test
                if len(self.data) == 0:
                    data_out_test = get_data(test_data_csv, data_point_limit=data_point_limit)
                    self.data = data_out_test
            elif split == 'train':
                global data_out_train
                self.data = data_out_train
                if len(self.data) == 0:
                    data_out_train = get_data(train_data_csv, data_point_limit=data_point_limit)
                    self.data = data_out_train
            else:
                raise ValueError("Split must be 'test' or 'train'")
        else:
            self.data = data
        self.processed_data = []
        for datum in self.data:
            x_full, y = datum
            y = torch.Tensor(int(y))
            if x_full is None:
                raise ValueError("x_video_full is None")
            x = get_window_npy(x_full, syncnet_T=frames, start_id=0)
            if x is not None:
                self.processed_data.append((x, y))
        
    def __len__(self):
        return len(self.processed_data)
    def __getitem__(self, idx):
        return self.processed_data[idx]

def get_lmk_feat(dataloader, st_gcn_model, device='cpu', strip_z=True):
    y_truth = []
    v_vals = []
    v_still_vals = []
    with torch.no_grad():
        prog_bar = enumerate(dataloader)
        for step, (x, y) in prog_bar:

            x = x.permute(0, 3, 1, 2)
            if strip_z:
                x = x[:, :2, :, :]

            lmk_feat = st_gcn_model(x)
            v = lmk_feat.mean(dim=1)
            v_vals.append(v)

            temporal_dim = 2
            num_frames = x.shape[temporal_dim]
            x_still = np.repeat(x[:, :, 0:1, :], repeats=num_frames, axis=temporal_dim)

            still_feat = st_gcn_model(x_still)
            v_still = still_feat.mean(dim=1)
            v_still_vals.append(v_still)

            y_truth.extend(y.view(-1).detach().cpu().tolist())
    return y_truth, v_vals, v_still_vals



if __name__ == "__main__":
    data_limit = None
    use_cuda = torch.cuda.is_available()
    data_limit = 10
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 1 # hparams.syncnet_batch_size
    num_workers = 1

    test_dataset = Dataset_Frames('test', data_point_limit=data_limit)
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=num_workers)
    

    edges = list(FACEMESH_TESSELATION)
    V = 478
    A = build_adjacency(V, edges)

    st_gcn_model, audio_model = st.load_stgcn_and_audio_models(st_gcn_checkpoint, audio_checkpoint, A, V, use_cuda)

    get_lmk_feat(test_data_loader, st_gcn_model, device=device)