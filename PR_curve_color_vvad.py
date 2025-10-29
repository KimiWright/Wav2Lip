from color_syncnet_eval import computeDist, model, babble_mel, silent_mel, white_noise_mel
# import run_statistics as run_stats

import torch
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import os


source_main_path = "/home/ksw38/.cache/kagglehub/datasets/adrianlubitz/vvadlrs3/versions/4/faceImages_small.h5"
use_cuda = False
data_limit = 10
# use_cuda = torch.cuda.is_available()
# data_limit = None


device = "cuda" if use_cuda else "cpu"
model.to(device)


print(device)


def get_losses_and_fconfms(mel, x):
    # print(next(model.parameters()).device, mel.device, x.device)
    a, v = model(mel, x)
    loss = F.cosine_similarity(a, v)
    fconfm = computeDist(a, v, vshift=15)
    return loss, fconfm

syncnet_T = 5
start_frame_num = 0
comp_names = ["silence", "white_noise", "babble_noise", "still_face"]
# babble_mel = run_stats.generate_babble_mel()
# silent_mel = run_stats.generate_mel_for_frames(syncnet_T, silence=True)
# white_noise_mel = run_stats.generate_mel_for_frames(syncnet_T, silence=False)
comp_mels = [silent_mel, white_noise_mel, babble_mel]
num_comp = len(comp_names)

def get_x(frames):
    x = np.concatenate(frames, axis=2)/255
    x = x.transpose(2, 0, 1)
    x = x[:, x.shape[1]//2:]
    x = torch.FloatTensor(x)
    x = x.unsqueeze(0)  # Add batch dimension
    return x


def get_all_results(x_files, y_files):
    losses = [[] for _ in range(num_comp)]
    fconfms = [[] for _ in range(num_comp)]
    y_vals = []

    for i, frames in enumerate(x_files):
        frames = frames[start_frame_num:start_frame_num+syncnet_T] ## Full Video
        y = torch.FloatTensor([y_files[i]]).to(device).unsqueeze(0)  # Convert to tensor and add batch dimension
        y_vals.append(y_files[i])

        x = get_x(frames).to(device)

        for j, mel in enumerate(comp_mels):
            mel = torch.FloatTensor(mel).to(device)
            loss, fconfm = get_losses_and_fconfms(mel, x)
            losses[j].append(loss.item())
            fconfms[j].append(fconfm.item())

        still_face_frames = frames[0:1].repeat(syncnet_T, axis=0)
        still_x = get_x(still_face_frames).to(device)
        mel = torch.FloatTensor(comp_mels[0]).to(device)
        still_a, still_v = model(mel, still_x) # mel doesn't matter
        a, v = model(mel, x)
        loss = F.cosine_similarity(still_v, v)
        fconfm = computeDist(still_v, v, vshift=15)

        losses[num_comp-1].append(loss.item())
        fconfms[num_comp-1].append(fconfm.item())

        if data_limit is not None and i > data_limit:
            break

    return y_vals, losses, fconfms

def fig_path_and_title(name, folder="VVAD_PR_Curves/color"):
    fig_path = os.path.join(folder, name+'.png')
    fig_title = f"PR curve for {name} VVAD"
    return fig_path, fig_title

def plot_PR_curve(name, y_test, y_scores, save_csv=True):
    fig_path, fig_title = fig_path_and_title(name)
    print("curve")
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    print(len(precision), len(recall), len(thresholds))
    auc_score = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {auc_score:.2f})', drawstyle="steps-post")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(fig_title)
    plt.legend()
    plt.savefig(fig_path)
    print(f"Figure saved at {fig_path}")

    if save_csv:
        csv_path = os.path.splitext(fig_path)[0] + "_PR_data.csv"
        df = pd.DataFrame({
            "precision": precision,
            "recall": recall,
            # thresholds is 1 shorter than precision/recall
            "threshold": list(thresholds) + [None]
        })
        df.to_csv(csv_path, index=False)
        print(f"Precision-Recall data saved to {csv_path}")
    
    return(auc_score, precision, recall, thresholds)


if __name__ == "__main__":
    with h5py.File(source_main_path, 'r') as f:
        # Get frames from the h5 file
        x_test = f['x_test']
        x_train = f['x_train']
        # Get the ground truth labels
        y_test = f['y_test']
        y_train = f['y_train']

        y_vals, losses, fconfms = get_all_results(x_test, y_test)

        for i in range(num_comp):
            name = "color"+"_"+comp_names[i]
            print(name)
            print(len(losses[i]))
            print(len(fconfms[i]))
            plot_PR_curve(name + "_cosine", y_vals, losses[i])
            plot_PR_curve(name + "_fconfm", y_vals, fconfms[i])

        for i in range(num_comp):
            name = "color"+"_"+comp_names[i]+"_neg"
            plot_PR_curve(name + "_cosine", y_vals, -np.array(losses[i]))
            plot_PR_curve(name + "_fconfm", y_vals, -np.array(fconfms[i]))