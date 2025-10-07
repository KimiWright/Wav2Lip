import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as data_utils

from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

import color_syncnet_train as train
from models import SyncNet_color as SyncNet
from hparams import hparams

# ----------------------
# Config
# ----------------------
fig_path        = "PR_curve_margin_color_personally_trained.png"
eval_step_max   = None          # e.g., 1000 to cap eval examples
check_in_steps  = 100
max_shift       = 15            # frames to search left/right for alignment
checkpoint_path = train.args.checkpoint_path

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

# ----------------------
# Scoring: SyncNet-style margin score
#   score = sim(shift=0) - max_{shift!=0} sim(shift)
# where sim is mean cosine similarity over the *overlapping* timesteps
# ----------------------
def sync_margin_score(a_seq: torch.Tensor, v_seq: torch.Tensor, max_shift: int = 15) -> float:
    """
    a_seq, v_seq: [T, D] audio/video embedding sequences for a single example.
    Returns a higher-is-better margin indicating that zero-shift looks best.
    """
    # Ensure [T, D]
    if a_seq.dim() == 1: a_seq = a_seq.unsqueeze(0)
    if v_seq.dim() == 1: v_seq = v_seq.unsqueeze(0)
    assert a_seq.dim() == 2 and v_seq.dim() == 2, f"Expected [T, D]; got {a_seq.shape}, {v_seq.shape}"

    T = min(a_seq.size(0), v_seq.size(0))
    if T == 0:
        return float("-inf")

    a_seq = a_seq[:T]
    v_seq = v_seq[:T]

    sims = []
    has_nonzero = False
    for s in range(-max_shift, max_shift + 1):
        if s < 0:
            # audio is *ahead* of video by |s|
            a_aligned = a_seq[-s:]        # drop first |s| from audio
            v_aligned = v_seq[:T + s]     # drop last |s| from video
        elif s > 0:
            # video is ahead of audio by s
            a_aligned = a_seq[:T - s]
            v_aligned = v_seq[s:]
        else:
            a_aligned = a_seq
            v_aligned = v_seq

        if a_aligned.size(0) == 0:
            sims.append(float("-inf"))
            continue

        sim = F.cosine_similarity(a_aligned, v_aligned, dim=1).mean()
        sims.append(sim.item())
        if s != 0 and sims[-1] != float("-inf"):
            has_nonzero = True

    zero_sim = sims[max_shift]  # s = 0 is centered
    if not has_nonzero:
        # No valid non-zero overlaps (clip too short vs. max_shift) -> fall back to zero-sim
        return zero_sim

    nonzero_best = max(sims[:max_shift] + sims[max_shift + 1:])
    return zero_sim - nonzero_best

# ----------------------
# Evaluation loop
# ----------------------
def eval_model_syncnet_task(test_loader, device, model, max_shift=15, check_in_steps=100, eval_step_max=None):
    model.eval()
    y_true, y_score = [], []

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            # Expect (x, mel, y)
            x, mel, y = batch
            x   = x.to(device, non_blocking=True)
            mel = mel.to(device, non_blocking=True)
            y   = y.to(device, non_blocking=True)

            # Model returns embeddings for audio/video
            # Preferably shapes: [B, T, D]
            a, v = model(mel, x)

            # Make batch-safe; handle possible [B, D] by promoting to T=1
            if a.dim() == 2:  # [B, D] -> [B, 1, D]
                a = a.unsqueeze(1)
            elif a.dim() == 1:           # [D] -> [1,1,D]
                a = a.unsqueeze(0).unsqueeze(1)
            if v.dim() == 2:
                v = v.unsqueeze(1)
            elif v.dim() == 1:
                v = v.unsqueeze(0).unsqueeze(1)

            B = a.size(0)
            for b in range(B):
                a_seq = a[b]          # [T, D]
                v_seq = v[b]          # [T, D]
                score = sync_margin_score(a_seq, v_seq, max_shift=max_shift)
                y_score.append(float(score))

            # Collect labels as 0/1
            y_true.extend(y.view(-1).detach().float().cpu().tolist())

            if eval_step_max is not None and step >= eval_step_max:
                break

            if check_in_steps and step % check_in_steps == 0 and step > 0:
                print(f"[Eval] Step {step}: collected {len(y_true)} examples")

    return y_true, y_score

# ----------------------
# Main
# ----------------------
def main():
    model = SyncNet().to(device)
    print('total trainable params', sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=hparams.syncnet_lr)

    print(f"Loading checkpoint from: {checkpoint_path}")
    if checkpoint_path:
        train.load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False, use_cuda=use_cuda)

    test_dataset = train.Dataset('val')
    test_loader = data_utils.DataLoader(
        test_dataset,
        batch_size=1,            # keep 1 if your dataset yields full clips; raise if safe
        num_workers=1,
        pin_memory=use_cuda
    )

    y_true, y_score = eval_model_syncnet_task(
        test_loader, device, model,
        max_shift=max_shift,
        check_in_steps=check_in_steps,
        eval_step_max=eval_step_max
    )

    # Sanity: ensure binary labels {0,1}
    uniq = sorted(set(int(t) for t in y_true))
    if not set(uniq).issubset({0, 1}):
        raise ValueError(f"Labels must be 0/1; got {uniq}")

    # Metrics
    precision, recall, thresholds = precision_recall_curve(y_true, y_score, pos_label=1)
    ap  = average_precision_score(y_true, y_score)
    roc = roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else float("nan")

    # Best-F1 on PR curve
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1))
    best_f1 = float(f1[best_idx])
    best_thr = float(thresholds[max(0, best_idx - 1)]) if len(thresholds) else float("nan")

    print(f"Eval results: AP={ap:.4f}  ROC-AUC={roc:.4f}  Best-F1={best_f1:.4f} @ thr={best_thr:.4f}")

    # Plot PR
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('SyncNet (shift-margin) Precision–Recall')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    print(f"Saved PR curve to: {os.path.abspath(fig_path)}")

if __name__ == "__main__":
    main()
