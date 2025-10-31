import PR_curve_train_st_gcn_vvad as curve
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION 
import PR_curve_mediapipe_vvad as vvad

import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils import data as data_utils
from sklearn.metrics import accuracy_score

from models import build_adjacency

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

def get_logits(test_data_loader, vvad_model, strip_z=True, device="cpu"):
    all_logits = []
    y_vals = []

    for step, (x, y) in enumerate(test_data_loader):

        vvad_model.eval()

        x = x.permute(0, 3, 1, 2)
        # print(x.shape)
        if strip_z:
            # print("stripping")
            x = x[:, :2, :, :]
        x = x.to(device)
        # print(x.shape)

        logits = vvad_model(x)
        y = y.to(device)

        all_logits.append(logits.item())
        y_vals.append(y.item())
    return y_vals, all_logits

def plots_from_checkpoint(name, test_data_loader, vvad_checkpoint_dir, A, V, use_cuda, rotation, device):
    vvad_model = curve.load_vvad_model(vvad_checkpoint_dir, A, V, use_cuda, rotation)
    vvad_model = vvad_model.eval().to(device)

    y_vals, all_logits = get_logits(test_data_loader, vvad_model, strip_z=True, device=device)
    auc_score, precision, recall, thresholds = curve.plot_PR_curve(name, y_vals, all_logits)
    # curve.plot_PR_curve(name+" Reversed", y_vals, all_logits)
    best_accuracy_threshold, best_acc = best_accuracy(y_vals, all_logits, thresholds)
    print(best_acc)

if __name__ == "__main__":
    data_limit = 10
    use_cuda = False
    data_limit = None
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    batch_size = 1
    syncnet_T = 5
    
    test_dataset = vvad.Dataset_Frames("test", frames=syncnet_T, data_point_limit=data_limit)
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=1, shuffle=True)

    name = "Mediapipe"
    fig_path, fig_title = curve.fig_path_and_title(name)
    print(fig_path, fig_title)

    edges = list(FACEMESH_TESSELATION)
    V = 478
    A = build_adjacency(V, edges)

    vvad_checkpoint_dir = "ckpt_folder/checkpoints_vvad_mediapipe"

    plot_args = dict(
            name=name, 
            test_data_loader=test_data_loader, 
            vvad_checkpoint_dir = vvad_checkpoint_dir, 
            A = A, 
            V = V, 
            use_cuda=use_cuda, 
            rotation=False, 
            device = device
        )
    plots_from_checkpoint(**plot_args)