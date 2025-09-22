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
from hparams import hparams
import st_gcn_test as st

from models import audio_only
from models import LandmarkSTGCNConformerWithOrientation
from models import LandmarkSTGCNConformer
from models import build_adjacency

### Variables ###

eval_step_max = None
# use_cuda = False
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

norot_knn_check = "checkpoint_step000330000.pth"
norot_check = "checkpoint_step000250000.pth"
rot_check = "checkpoint_step000240000.pth"
## ST GCN
st_gcn_norot_checkpoint_knn = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints_st_gcn_norot/" + norot_knn_check
st_gcn_norot_checkpoint = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints_st_gcn_norot_facial/" + norot_check
st_gcn_rot_checkpoint = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints_st_gcn_rot_facial/" + rot_check

## Audio
audio_norot_checkpoint_knn = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints_audio_norot/" + norot_knn_check
audio_norot_checkpoint = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints_audio_norot_facial/" + norot_check
audio_rot_checkpoint = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints_audio_rot_facial/" + rot_check

### Functions ###
def get_checkpoint(checkpoint):
    if os.path.isdir(checkpoint):
        checkpoint_path = os.listdir(checkpoint)[-1]
        checkpoint_path = os.path.join(checkpoint, checkpoint_path)
    else:
        checkpoint_path = checkpoint
    return checkpoint_path

def load_from_checkpoint_or_dir(checkpoint, model, optimizer, reset_optimizer=False, use_cuda=False):
    checkpoint_path = get_checkpoint(checkpoint)
    st.load_checkpoint(checkpoint_path, model, optimizer=optimizer, reset_optimizer=reset_optimizer, use_cuda=use_cuda)
    return model

def fig_path_and_title(model_type):
    fig_path = f"PR_curve_lmks_{model_type}.png"
    fig_title = f"Precision-Recall for {model_type} on determining if audio and video are synced"
    return fig_path, fig_title

def plot_PR_curve(model_type, y_test, y_scores):
    fig_path, fig_title = fig_path_and_title(model_type)
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    auc_score = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {auc_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(fig_title)
    plt.legend()
    plt.savefig(fig_path)


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

    print(f"\t and audio model from {audio_norot_checkpoint}")
    audio_model = audio_only().to(device)
    audio_optimizer = optim.Adam([p for p in audio_model.parameters() if p.requires_grad],
                                lr=hparams.syncnet_lr, weight_decay=1e-5)
    audio_model = load_from_checkpoint_or_dir(audio_checkpoint, model=audio_model, optimizer=audio_optimizer, use_cuda=use_cuda)
    audio_model.eval()

    print('total trainable params for stgcn: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('total trainable params for audio: {}'.format(sum(p.numel() for p in audio_model.parameters() if p.requires_grad)))
    return model, audio_model

### Eval Loop ###

def eval_norot_model(test_data_loader, st_gcn_model, audio_model, device):
    eval_steps = eval_step_max
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    y_truth = []
    for step, (x, x_rot, mel, y) in enumerate(test_data_loader):

        st_gcn_model.eval()
        audio_model.eval()

        # Transform data to CUDA device

        mel = mel.to(device)

        x = x.permute(0, 2, 1, 3).to(device)

        lmk_feat = st_gcn_model(x)
        v = lmk_feat.mean(dim=1)
        a = audio_model(mel)
        y = y.to(device)

        loss = F.cosine_similarity(a, v)
        losses.extend(loss.view(-1).detach().cpu().tolist())
        y_truth.extend(y.view(-1).detach().cpu().tolist())

        if eval_steps is not None and step > eval_steps: break 

    averaged_loss = sum(losses) / len(losses)
    print(averaged_loss)

    return y_truth, losses

def eval_rot_model(test_data_loader, st_gcn_model, audio_model, device):
    eval_steps = eval_step_max
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    y_truth = []
    for step, (x, x_rot, mel, y) in enumerate(test_data_loader):

        st_gcn_model.eval()
        audio_model.eval()

        # Transform data to CUDA device

        mel = mel.to(device)

        x = x.permute(0, 2, 1, 3).to(device)
        x_rot = x_rot.permute(0, 2, 1).to(device)

        lmk_feat = st_gcn_model(x, x_rot)
        v = lmk_feat.mean(dim=1)
        a = audio_model(mel)
        y = y.to(device)

        loss = F.cosine_similarity(a, v)
        losses.extend(loss.view(-1).detach().cpu().tolist())
        y_truth.extend(y.view(-1).detach().cpu().tolist())

        if eval_steps is not None and step > eval_steps: break ## Modification ##

    averaged_loss = sum(losses) / len(losses)
    print(averaged_loss)

    return y_truth, losses

if __name__ == "__main__":
    ### Set Up ###
    batch_size = 1 # hparams.syncnet_batch_size
    test_dataset = st.Dataset('val')

    device = "cuda" if use_cuda else "cpu"
    print(f"Using {device}")

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=8)

    first_point = test_dataset[0]
    (x, x_rot, mel, y) = first_point
    first_lmks = x[0].T
    knn_edges = st.knn_edges(first_lmks)
    facial_edges = st.facial_edges()
    num_lmks = first_lmks.shape[0]

    print(num_lmks)

    A = build_adjacency(num_lmks, facial_edges)
    A_knn = build_adjacency(num_lmks, knn_edges)
    V = num_lmks

    ### Init models ###
    st_gcn_model_norot, audio_model_norot = load_stgcn_and_audio_models(st_gcn_norot_checkpoint, audio_norot_checkpoint, A, V, use_cuda=use_cuda, rotation=False)
    st_gcn_model_norot_knn, audio_model_norot_knn = load_stgcn_and_audio_models(st_gcn_norot_checkpoint_knn, audio_norot_checkpoint_knn, A_knn, V, use_cuda=use_cuda, rotation=False)
    st_gcn_model_rot, audio_model_rot = load_stgcn_and_audio_models(st_gcn_rot_checkpoint, audio_rot_checkpoint, A, V, use_cuda=use_cuda, rotation=True)

    ### Create PR Curve
    y_truth, scores = eval_norot_model(test_data_loader, st_gcn_model_norot, audio_model_norot, device)
    plot_PR_curve("ST_GCN without Rotation", y_truth, scores)

    y_truth, scores = eval_norot_model(test_data_loader, st_gcn_model_norot_knn, audio_model_norot_knn, device)
    plot_PR_curve("ST_GCN without Rotation using Knn", y_truth, scores)

    y_truth, scores = eval_rot_model(test_data_loader, st_gcn_model_rot, audio_model_rot, device)
    plot_PR_curve("ST_GCN with Rotation", y_truth, scores)
