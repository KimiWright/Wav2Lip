from models import LandmarkSTGCNConformer, LandmarkSTGCNConformerWithOrientation, build_adjacency, STGCNConformerVVAD
import PR_curve_mediapipe_vvad as vvad
from hparams import hparams
import st_gcn_train_mediapipe as st

from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION 
from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
from torch.utils import data as data_utils
from os.path import join

global_step = 0
global_epoch = 0
strip_z = True

criterion = nn.BCEWithLogitsLoss()

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def load_stgcn(checkpoint, A, V, use_cuda = False):
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
    
    model = LandmarkSTGCNConformer(**model_args)
    model.to(device)

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                            lr=hparams.syncnet_lr, weight_decay=1e-5)
    print('total trainable params for stgcn: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model = st.load_from_checkpoint_or_dir(checkpoint, model=model, optimizer=optimizer, use_cuda=use_cuda)

    print('total trainable params for stgcn: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    return model

def eval_model(test_data_loader, device, vvad_model):
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (x, y) in enumerate(test_data_loader):
            vvad_model.eval()

            x = x.permute(0, 3, 1, 2)
            if strip_z:
                x = x[:, :2, :, :]
            x = x.to(device)

            logits = vvad_model(x)
            y = y.to(device)

            loss = criterion(logits.squeeze(), y.squeeze().float())
            losses.append(loss.item())

            if eval_steps is not None and step > eval_steps: break

        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)

        return


def train(device, vvad_model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step

    while global_epoch < nepochs:
        running_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, y) in prog_bar:
            vvad_model.train()
            optimizer.zero_grad()

            x = x.permute(0, 3, 1, 2)
            if strip_z:
                x = x[:, :2, :, :]
            x = x.to(device)

            logits = vvad_model(x)
            y = y.to(device)
            # print(y, y.shape, y.unsqueeze(0).shape)

            loss = criterion(logits.squeeze(), y.squeeze().float())
            loss.backward()
            optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    vvad_model, optimizer, global_step, checkpoint_dir, global_epoch)
            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    eval_model(test_data_loader, device, vvad_model)



if __name__ == "__main__":
    batch_size = 1
    data_limit = None
    batch_size = hparams.syncnet_batch_size
    syncnet_T = 5
    

    vvad_checkpoint_dir = "/home/ksw38/RVL/color_syncnet/Wav2Lip/ckpt_folder/checkpoints_vvad_mediapipe"
    st_gcn_checkpoint_dir = "/home/ksw38/RVL/color_syncnet/Wav2Lip/ckpt_folder/checkpoints_mediapipe"

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"Using {device}")

    test_dataset = vvad.Dataset_Frames("test", frames=syncnet_T, data_point_limit=data_limit)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=1, shuffle=True)
    
    train_dataset = vvad.Dataset_Frames("train", frames=syncnet_T, data_point_limit=data_limit)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=1, shuffle=True)

    edges = list(FACEMESH_TESSELATION)
    V = 478
    A = build_adjacency(V, edges)

    st_gcn_model = load_stgcn(st_gcn_checkpoint_dir, A, V, use_cuda)
    global_epoch = st.global_epoch
    global_step = st.global_step
    vvad_model = STGCNConformerVVAD(st_gcn_model)
    vvad_model.eval().to(device)

    vvad_optimizer = optim.Adam([p for p in vvad_model.parameters() if p.requires_grad],
                            lr=hparams.syncnet_lr, weight_decay=1e-5)
    
    # Training
    print("Beginning Training")
    train(device, vvad_model, train_data_loader, test_data_loader, vvad_optimizer, vvad_checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)


