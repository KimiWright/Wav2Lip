from color_syncnet_eval import computeDist, model, babble_mel, silent_mel, white_noise_mel
# import run_statistics as run_stats

import torch
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os




source_main_path = "/home/ksw38/.cache/kagglehub/datasets/adrianlubitz/vvadlrs3/versions/4/faceImages_small.h5"
use_cuda = False
device = "cuda" if use_cuda else "cpu"
data_limit = 10

print(device)


def get_losses_and_fconfms(mel, x):
    a, v = model(mel, x)
    loss = F.cosine_similarity(a, v)
    fconfm = computeDist(a, v, vshift=15)
    return loss, fconfm

syncnet_T = 5
start_frame_num = 0
comp_names = ["still_face", "silence", "white_noise", "babble_noise"]
# babble_mel = run_stats.generate_babble_mel()
# silent_mel = run_stats.generate_mel_for_frames(syncnet_T, silence=True)
# white_noise_mel = run_stats.generate_mel_for_frames(syncnet_T, silence=False)
comp_mels = [silent_mel, white_noise_mel, babble_mel]
num_comp_mel = len(comp_mels)

def get_all_results(x_files, y_files):
    losses = [[] for _ in range(num_comp_mel)]
    fconfms = [[] for _ in range(num_comp_mel)]
    y_vals = []

    for i, frames in enumerate(x_files):
        frames = frames[start_frame_num:start_frame_num+syncnet_T] ## Full Video
        y = torch.FloatTensor([y_files[i]]).to(device).unsqueeze(0)  # Convert to tensor and add batch dimension
        y_vals.append(y_files[i])

        x = np.concatenate(frames, axis=2)/255
        x = x.transpose(2, 0, 1)
        x = x[:, x.shape[1]//2:]
        x = torch.FloatTensor(x)
        x = x.unsqueeze(0)  # Add batch dimension
        x = x.to(device)

        for i, mel in enumerate(comp_mels):
            loss, fconfm = get_losses_and_fconfms(mel, x)
            losses[i].append(loss.item())
            fconfms[i].append(fconfm.item())

        if data_limit is not None and i > data_limit:
            break

    return losses, fconfms

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
    return(auc_score, precision, recall, thresholds)


if __name__ == "__main__":
    with h5py.File(source_main_path, 'r') as f:
        # Get frames from the h5 file
        x_test = f['x_test']
        x_train = f['x_train']
        # Get the ground truth labels
        y_test = f['y_test']
        y_train = f['y_train']

        losses, fconfms = get_all_results(x_test, y_test)
        
        print(len(losses))
        print(len(fconfms))

        for i in range(num_comp_mel):
            name = "color"+"_"+comp_names[i+1]
            print(name)