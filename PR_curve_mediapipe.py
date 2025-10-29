import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils import data as data_utils
from hparams import hparams
import st_gcn_train_mediapipe as st
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION 

from models import audio_only
from models import LandmarkSTGCNConformer
from models import build_adjacency

st_gcn_checkpoint = st.args.st_gcn_checkpoint_dir
audio_checkpoint = st.args.audio_checkpoint_dir

fig_path = f"Syncnet_task_PR_Curves/PR_curve_lmks_mediapipe.png"
fig_title = f"Precision-Recall for mediapipe landmarks model on determining if audio and video are synced"
eval_step_max = None

def eval_model(test_data_loader, device, st_gcn_model, audio_model, strip_z = True):
    eval_steps = eval_step_max
    # eval_steps = None ## Modification ##
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    y_truth = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):

            st_gcn_model.eval()
            audio_model.eval()

            # Transform data to CUDA device

            mel = mel.to(device)

            x = x.permute(0, 3, 1, 2)
            if strip_z:
                x = x[:, :2, :, :]

            lmk_feat = st_gcn_model(x)
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
    
def plot_PR_curve(fig_path, fig_title, y_test, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    auc_score = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {auc_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(fig_title)
    plt.legend()
    plt.savefig(fig_path)

    print(f"Saved PR Curve at {fig_path}")

    csv_path = os.path.splitext(fig_path)[0] + "_PR_data.csv"
    df = pd.DataFrame({
        "precision": precision,
        "recall": recall,
        # thresholds is 1 shorter than precision/recall
        "threshold": list(thresholds) + [None]
    })
    df.to_csv(csv_path, index=False)
    print(f"Precision-Recall data saved to {csv_path}")
    
if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 1 # hparams.syncnet_batch_size
    test_dataset = st.Dataset('val')

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=1)
    
    edges = list(FACEMESH_TESSELATION)
    V = 478
    A = build_adjacency(V, edges)

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

    print(audio_checkpoint)
    
    st_gcn_model, audio_model = st.load_stgcn_and_audio_models(st_gcn_checkpoint, audio_checkpoint, A, V, use_cuda)

    y_truth, scores = eval_model(test_data_loader, device, st_gcn_model, audio_model)
    print(len(y_truth))
    print(len(scores))

    plot_PR_curve(fig_path, fig_title, y_truth, scores)
