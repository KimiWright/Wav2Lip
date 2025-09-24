import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils import data as data_utils
import numpy as np

from hparams import hparams
import st_gcn_test as st
import st_gcn_vvad as vvad
import PR_curve_st_gcn as pr
import run_statistics as run_stats

from models import audio_only
from models import LandmarkSTGCNConformerWithOrientation
from models import LandmarkSTGCNConformer
from models import build_adjacency

### Variables ###
# use_cuda = torch.cuda.is_available()
use_cuda = False
syncnet_T = 5
device = torch.device("cuda" if use_cuda else "cpu")

norot_knn_check = ""# "checkpoint_step000330000.pth"
norot_check = "checkpoint_step000440000.pth" # "checkpoint_step000250000.pth"
rot_check = "checkpoint_step000410000.pth" # "checkpoint_step000240000.pth"
## ST GCN
st_gcn_norot_checkpoint_knn = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints_st_gcn_norot/" + norot_knn_check
st_gcn_norot_checkpoint = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints_st_gcn_norot_facial/" + norot_check
st_gcn_rot_checkpoint = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints_st_gcn_rot_facial/" + rot_check

## Audio
audio_norot_checkpoint_knn = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints_audio_norot/" + norot_knn_check
audio_norot_checkpoint = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints_audio_norot_facial/" + norot_check
audio_rot_checkpoint = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints_audio_rot_facial/" + rot_check

comp_names = ["still_face", "silence", "white_noise", "babble_noise"]
babble_mel = run_stats.generate_babble_mel()
silent_mel = run_stats.generate_mel_for_frames(syncnet_T, silence=True)
white_noise_mel = run_stats.generate_mel_for_frames(syncnet_T, silence=False)
comp_mels = [silent_mel, white_noise_mel, babble_mel]


def get_lmk_feat_norot(test_data_loader, st_gcn_model, device='cpu'):
    y_truth = []
    v_vals = []
    v_still_vals = []
    with torch.no_grad():
        prog_bar = enumerate(test_data_loader)
        for step, (x, x_rot, y) in prog_bar:
            st_gcn_model.eval()

            x = x.permute(0, 2, 1, 3).to(device)

            lmk_feat = st_gcn_model(x)
            v = lmk_feat.mean(dim=1)
            y = y.to(device)
            
            y_truth.extend(y.view(-1).detach().cpu().tolist())
            v_vals.append(v)

            temporal_dim = 2
            num_frames = x.shape[temporal_dim]
            x_still = np.repeat(x[:, :, 0:1, :], repeats=num_frames, axis=temporal_dim)

            still_feat = st_gcn_model(x_still)
            v_still = still_feat.mean(dim=1)
            v_still_vals.append(v_still)
    return y_truth, v_vals, v_still_vals

def get_lmk_feat_rot(test_data_loader, st_gcn_model, device='cpu'):
    y_truth = []
    v_vals = []
    v_still_vals = []
    with torch.no_grad():
        prog_bar = enumerate(test_data_loader)
        for step, (x, x_rot, y) in prog_bar:
            st_gcn_model.eval()

            x = x.permute(0, 2, 1, 3).to(device)
            x_rot = x_rot.permute(0, 2, 1).to(device)

            lmk_feat = st_gcn_model(x, x_rot)
            v = lmk_feat.mean(dim=1)
            y = y.to(device)
            
            y_truth.extend(y.view(-1).detach().cpu().tolist())
            v_vals.append(v)

            temporal_dim = 2
            num_frames = x.shape[temporal_dim]
            x_still = np.repeat(x[:, :, 0:1, :], repeats=num_frames, axis=temporal_dim)
            x_rot_still = np.repeat(x_rot[:, 0:1, :], repeats=num_frames, axis=1)
            
            still_feat = st_gcn_model(x_still, x_rot_still)
            v_still = still_feat.mean(dim=1)
            v_still_vals.append(v_still)
    return y_truth, v_vals, v_still_vals

def eval_still_face(v_vals, v_still_vals):
    num_vals = len(v_vals)
    losses = []
    if num_vals != len(v_still_vals):
        raise(ValueError(f"The lengths of v and v_still are different! v_vals: {num_vals}, v_still_vals: {len(v_still_vals)}"))
    for i in range(num_vals):
        loss = F.cosine_similarity(v_vals[i], v_still_vals[i])
        losses.extend(loss.view(-1).detach().cpu().tolist())
    return np.array(losses)

def eval_mel(v_vals, audio_model, comp_mel, device):
    batch_size = v_vals[0].shape[0]
    mel = comp_mel.expand(batch_size, -1, -1, -1).to(device)
    audio_model.eval()
    with torch.no_grad():
        a = audio_model(mel)
    losses = []
    for v in v_vals:
        loss = F.cosine_similarity(a, v)
        losses.extend(loss.view(-1).detach().cpu().tolist())
    return np.array(losses)

def fig_path_and_title(name, folder="VVAD_PR_Curves"):
    fig_path = os.path.join(folder, name+'.png')
    fig_title = f"PR curve for {name} VVAD"
    return fig_path, fig_title

def plot_PR_curve(name, y_test, y_scores):
    fig_path, fig_title = fig_path_and_title(name)
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    auc_score = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {auc_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(fig_title)
    plt.legend()
    plt.savefig(fig_path)
    print(f"Figure saved at {fig_path}")
    return(auc_score)

def eval_and_plot(st_gcn_checkpoint, audio_checkpoint, A, V, model_name, use_cuda=use_cuda, rotation=False):
    if rotation:
        st_gcn_model, audio_model = pr.load_stgcn_and_audio_models(st_gcn_checkpoint, audio_checkpoint, A, V, use_cuda=use_cuda, rotation=rotation)
        y_truth, v_vals, v_still_vals = get_lmk_feat_rot(test_data_loader, st_gcn_model, device)
    else:
        st_gcn_model, audio_model = pr.load_stgcn_and_audio_models(st_gcn_checkpoint, audio_checkpoint, A, V, use_cuda=use_cuda, rotation=False)
        y_truth, v_vals, v_still_vals = get_lmk_feat_norot(test_data_loader, st_gcn_model, device)
    all_losses = []
    all_losses.append(eval_still_face(v_vals, v_still_vals))

    for comp_mel in comp_mels:
        all_losses.append(eval_mel(v_vals, audio_model, comp_mel, device))

    for i, losses in enumerate(all_losses):
        name = model_name+'_'+comp_names[i]
        auc_score = plot_PR_curve(name, y_truth, losses)
        print(f"AUC: {auc_score}\n")

    for i, losses in enumerate(all_losses):
        name = model_name+'_'+comp_names[i]+'_neg'
        auc_score = plot_PR_curve(name, y_truth, -losses)
        print(f"AUC: {auc_score}\n")

if __name__ == "__main__":
    batch_size = 1
    data_limit = None
    test_dataset = vvad.Dataset_Frames("test", frames=syncnet_T, data_point_limit=data_limit)
    print(len(test_dataset))
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=1, shuffle=True)

    first_point = test_dataset[0]
    (x, x_rot, y) = first_point
    first_lmks = x[0].T
    knn_edges = st.knn_edges(first_lmks)
    facial_edges = st.facial_edges()
    num_lmks = first_lmks.shape[0]

    A = build_adjacency(num_lmks, facial_edges)
    A_knn = build_adjacency(num_lmks, knn_edges)
    V = num_lmks

    eval_and_plot(st_gcn_norot_checkpoint, audio_norot_checkpoint, A, V, model_name="norot_facial", use_cuda=use_cuda)
    eval_and_plot(st_gcn_norot_checkpoint_knn, audio_norot_checkpoint_knn, A_knn, V, model_name="norot_knn", use_cuda=use_cuda)
    eval_and_plot(st_gcn_rot_checkpoint, audio_rot_checkpoint, A, V, "rot", use_cuda=use_cuda, rotation=True)

