from models import LandmarkSTGCNConformer, LandmarkSTGCNConformerWithOrientation, build_adjacency, STGCNConformerVVAD
import vvad_st_gcn_model_functions as st
import st_gcn_vvad as vvad
from hparams import hparams

from sklearn.metrics import precision_recall_curve, auc, accuracy_score
from torch.utils import data as data_utils
import matplotlib.pyplot as plt
from torch import optim
import numpy as np
import pandas as pd
import os

def best_accuracy(y_test, y_scores, thresholds):
    accuracies = []
    for thr in thresholds:
        preds = (y_scores >= thr).astype(int)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)

    best_acc_idx = max(range(len(accuracies)), key=lambda i: accuracies[i])
    best_acc_threshold = thresholds[best_acc_idx]
    best_acc = accuracies[best_acc_idx]

    return best_acc_threshold, best_acc

def load_vvad_model(checkpoint_dir, A, V, use_cuda=False, rotation=True):
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
        st_gcn_model = LandmarkSTGCNConformerWithOrientation(**model_args)
    else:
        st_gcn_model = LandmarkSTGCNConformer(**model_args)
    vvad_model = STGCNConformerVVAD(st_gcn_model)

    vvad_optimizer = optim.Adam([p for p in vvad_model.parameters() if p.requires_grad],
                            lr=hparams.syncnet_lr, weight_decay=1e-5)
    
    vvad_model = st.load_from_checkpoint_or_dir(checkpoint_dir, vvad_model, vvad_optimizer, use_cuda=use_cuda)
    print('total trainable params for stgcn: {}'.format(sum(p.numel() for p in vvad_model.parameters() if p.requires_grad)))
    return vvad_model

def get_logits(test_data_loader, vvad_model, rotation, device="cpu"):
    all_logits = []
    y_vals = []

    for step, (x, x_rot, y) in enumerate(test_data_loader):

        vvad_model.eval()

        x = x.permute(0, 2, 1, 3).to(device)
        x_rot = x_rot.permute(0, 2, 1).to(device)

        if rotation:
            logits = vvad_model(x, x_rot)
        else:
            logits = vvad_model(x)
        y = y.to(device)

        all_logits.append(logits.item())
        y_vals.append(y.item())

    return y_vals, all_logits

def fig_path_and_title(name, folder="VVAD_PR_Curves/trained"):
    fig_path = os.path.join(folder, name+'.png')
    fig_title = f"PR curve for {name} VVAD"
    return fig_path, fig_title

def plot_PR_curve(name, y_test, y_scores, save_csv=True):
    fig_path, fig_title = fig_path_and_title(name)
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
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

def plots_from_checkpoint(name, test_data_loader, vvad_checkpoint_dir, A, V, use_cuda, rotation, device):
    vvad_model = load_vvad_model(vvad_checkpoint_dir, A, V, use_cuda, rotation)
    vvad_model = vvad_model.eval().to(device)

    y_vals, all_logits = get_logits(test_data_loader, vvad_model, rotation, device)
    auc_score, precision, recall, thresholds = plot_PR_curve(name, y_vals, all_logits)
    # auc_score, precision, recall, thresholds = plot_PR_curve(name+" Reversed", y_vals, all_logits)
    best_accuracy_threshold, best_acc = best_accuracy(y_vals, all_logits, thresholds)
    print(best_acc)

if __name__ == "__main__":
    data_limit = None
    use_cuda = False
    device = "cuda" if use_cuda else "cpu"
    batch_size = 1
    syncnet_T = 5
    rot_facial_vvad_checkpoint_dir = "/home/ksw38/RVL/color_syncnet/Wav2Lip/ckpt_folder/checkpoints_vvad_rot_facial_st_gcn"
    rot_knn_vvad_checkpoint_dir = "/home/ksw38/RVL/color_syncnet/Wav2Lip/ckpt_folder/checkpoints_vvad_rot_st_gcn"
    norot_facial_vvad_checkpoint_dir = "/home/ksw38/RVL/color_syncnet/Wav2Lip/ckpt_folder/checkpoints_vvad_norot_facial_st_gcn"
    norot_knn_vvad_checkpoint_dir = "/home/ksw38/RVL/color_syncnet/Wav2Lip/ckpt_folder/checkpoints_vvad_norot_st_gcn"

    test_dataset = vvad.Dataset_Frames("test", frames=syncnet_T, data_point_limit=data_limit)
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=1, shuffle=True)
    
    facial_edges = st.facial_edges()

    first_point = test_dataset[0]
    (x, x_rot, y) = first_point
    first_lmks = x[0].T
    knn_edges = st.knn_edges(first_lmks)

    V_norm = 92

    A_facial_norm = build_adjacency(V_norm, facial_edges)
    A_knn_norm = build_adjacency(V_norm, knn_edges)
    
    name = "Facial Landmarks with Orientation"
    plot_args = dict(
            name=name, 
            test_data_loader=test_data_loader, 
            vvad_checkpoint_dir = rot_facial_vvad_checkpoint_dir, 
            A = A_facial_norm, 
            V = V_norm, 
            use_cuda=use_cuda, 
            rotation=True, 
            device = device
        )
    plots_from_checkpoint(**plot_args)


    name = "Knn Landmarks with Orientation"
    plot_args['name'] = name
    plot_args["vvad_checkpoint_dir"] = rot_knn_vvad_checkpoint_dir
    plot_args['A'] = A_knn_norm

    plots_from_checkpoint(**plot_args)

    name = "Facial Landmarks without Orientation"
    plot_args['name'] = name
    plot_args["vvad_checkpoint_dir"] = norot_facial_vvad_checkpoint_dir
    plot_args['A'] = A_facial_norm
    plot_args['rotation'] = False

    plots_from_checkpoint(**plot_args)

    name = "Knn Landmarks without Orientation"
    plot_args['name'] = name
    plot_args["vvad_checkpoint_dir"] = norot_knn_vvad_checkpoint_dir
    plot_args['A'] = A_knn_norm

    plots_from_checkpoint(**plot_args)
    

