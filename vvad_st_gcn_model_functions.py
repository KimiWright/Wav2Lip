from models import LandmarkSTGCNConformer, LandmarkSTGCNConformerWithOrientation, build_adjacency, STGCNConformerVVAD
from hparams import hparams
import torch
from torch import optim
import numpy as np
import os

## Model Loading ##
global_step = 0
global_epoch = 0

def _load(checkpoint_path, use_cuda): ## Modification, added use_cuda
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False, use_cuda=False): ## Modification, added use_cuda
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path, use_cuda)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model

def get_checkpoint(checkpoint):
    if os.path.isdir(checkpoint):
        checkpoint_path = os.listdir(checkpoint)[-1]
        checkpoint_path = os.path.join(checkpoint, checkpoint_path)
    else:
        checkpoint_path = checkpoint
    return checkpoint_path

def load_from_checkpoint_or_dir(checkpoint, model, optimizer, reset_optimizer=False, use_cuda=False):
    checkpoint_path = get_checkpoint(checkpoint)
    load_checkpoint(checkpoint_path, model, optimizer=optimizer, reset_optimizer=reset_optimizer, use_cuda=use_cuda)
    return model

def load_stgcn(checkpoint, A, V, use_cuda = False, rotation = False):
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

    print('total trainable params for stgcn: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    return model

## Edges ##
def knn_edges(points_xy, k=4):
    """
    Build undirected edges by connecting each landmark
    to its k nearest neighbors.
    
    Args:
        points_xy: np.array [V, 2], landmark coordinates (x,y)
        k: number of neighbors to connect
    
    Returns:
        edges: list of (i, j) tuples
    """
    V = points_xy.shape[0]
    edges = set()
    for i, p in enumerate(points_xy):
        # distances from point i to all others
        dists = np.linalg.norm(points_xy - p, axis=1)
        # get indices of k nearest (skip self at index 0)
        nearest = np.argsort(dists)[1:k+1]
        for j in nearest:
            edges.add((i, j))
            edges.add((j, i))  # make undirected
    return list(edges)

RVL_FACEMESH_LEFT_EYEBROW = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
RVL_FACEMESH_LEFT_EYE = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
RVL_FACEMESH_LIPS = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
RVL_FACEMESH_RIGHT_EYE = [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]
RVL_FACEMESH_RIGHT_EYEBROW = [82, 83, 84, 85, 86, 87, 88, 89, 90, 91]

def facial_edges():
    edges = []
    for region in [RVL_FACEMESH_LEFT_EYE, RVL_FACEMESH_RIGHT_EYE,
               RVL_FACEMESH_LEFT_EYEBROW, RVL_FACEMESH_RIGHT_EYEBROW,
               RVL_FACEMESH_LIPS]:
        for i in range(len(region)-1):
            edges.append((region[i], region[i+1]))
        edges.append((region[-1], region[0])) 

    return edges

if __name__ == "__main__":
    st_gcn_checkpoint_dir = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints_st_gcn_norot_facial/"

    use_cuda = False
    device = "cuda" if use_cuda else "cpu"
    print(f"Using {device}")

    V = 92
    facial_edges = facial_edges()
    A = build_adjacency(V, facial_edges)

    st_gcn_model = load_stgcn(st_gcn_checkpoint_dir, A, V, use_cuda, rotation=False)
    vvad_model = STGCNConformerVVAD(st_gcn_model)