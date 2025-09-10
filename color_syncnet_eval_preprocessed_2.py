import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data as data_utils
from scipy import signal
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from models import SyncNet_color as SyncNet
from hparams import hparams
from preprocessed_dataset import Dataset

# ----------------------------
# Load checkpoint
# ----------------------------
def load_syncnet(path, device="cpu"):
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model = SyncNet().to(device)
    res = model.load_state_dict(state, strict=False)
    print("Loaded with missing:", res.missing_keys, "unexpected:", res.unexpected_keys)
    model.eval()
    return model

# ----------------------------
# Compute distance (Chung+Zisserman style)
# ----------------------------
def calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift*2+1
    feat2p = F.pad(feat2, (0,0,vshift,vshift))
    dists = []
    for i in range(len(feat1)):
        d = F.pairwise_distance(feat1[[i],:].repeat(win_size, 1),
                                feat2p[i:i+win_size,:])
        dists.append(d)
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
        kernel = fconf.shape[0] // 2 * 2 + 1
    else:
        kernel = 9
    return signal.medfilt(fconf, kernel_size=kernel)

# ----------------------------
# Evaluate on dataset
# ----------------------------
def eval_dataset(model, loader, device="cpu", step_limit=None):
    scores, labels = [], []
    with torch.no_grad():
        for step, (x, mel, y) in enumerate(loader):
            x, mel = x.to(device), mel.to(device)
            a, v = model(mel, x)  # audio, video embeddings

            fconfm = computeDist(a, v)
            score = -float(np.mean(fconfm))  # single scalar per clip

            scores.append(score)
            labels.append(int(y.item()))

            if step_limit is not None and step >= step_limit:
                break
    return np.array(scores), np.array(labels)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    checkpoint_path = "lipsync_expert.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    step_limit = None

    model = load_syncnet(checkpoint_path, device)

    # datasets
    test_dataset = Dataset(split=None)
    test_loader = data_utils.DataLoader(test_dataset,
                                        batch_size=1, num_workers=1)

    print("Running evaluation...")
    scores, labels = eval_dataset(model, test_loader, device, step_limit)

    # stats
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    print(f"ROC AUC: {auc:.4f}")

    # plot histogram of scores
    plt.hist(scores[labels==1], bins=30, alpha=0.5, label="Speaking")
    plt.hist(scores[labels==0], bins=30, alpha=0.5, label="Not Speaking")
    plt.legend()
    plt.xlabel("SyncNet confidence score")
    plt.ylabel("Count")
    plt.title(f"Distribution (AUC={auc:.3f})")
    plt_path = "/home/ksw38/RVL/color_syncnet/Wav2Lip/color_syncnet_eval_preprocessed_2.png"
    plt.savefig(plt_path)
