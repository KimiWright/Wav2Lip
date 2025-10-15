import vvad_st_gcn_model_functions as st
import st_gcn_vvad as vvad
from models import LandmarkSTGCNConformer, LandmarkSTGCNConformerWithOrientation, build_adjacency, STGCNConformerVVAD
from hparams import hparams

from os.path import join
from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
from torch.utils import data as data_utils

criterion = nn.BCEWithLogitsLoss()

# logits = model(x)
# loss = criterion(logits.squeeze(), y.float())
# loss.backward()
# optimizer.step()

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

def eval_model(test_data_loader, device, vvad_model):
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (x, x_rot, y) in enumerate(test_data_loader):

            vvad_model.eval()

            x = x.permute(0, 2, 1, 3).to(device)

            logits = vvad_model(x)
            y = y.to(device)

            loss = criterion(logits.squeeze(), y.float())
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
        for step, (x, x_rot, y) in prog_bar:
            vvad_model.train()
            optimizer.zero_grad()

            x = x.permute(0, 2, 1, 3).to(device)

            logits = vvad_model(x)
            y = y.to(device)

            loss = criterion(logits.squeeze(), y.float())
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
    facial = False
    if facial:
        st_gcn_checkpoint_dir = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints_st_gcn_norot_facial/"
        vvad_checkpoint_dir = "/home/ksw38/RVL/color_syncnet/Wav2Lip/ckpt_folder/checkpoints_vvad_norot_facial_st_gcn"
    else:
        st_gcn_checkpoint_dir = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints_st_gcn_norot/"
        vvad_checkpoint_dir = "/home/ksw38/RVL/color_syncnet/Wav2Lip/ckpt_folder/checkpoints_vvad_norot_st_gcn"

    use_cuda = False
    device = "cuda" if use_cuda else "cpu"
    print(f"Using {device}")

    V = 92
    # facial_edges = st.facial_edges()
    # A = build_adjacency(V, facial_edges)
    

    test_dataset = vvad.Dataset_Frames("test", frames=syncnet_T, data_point_limit=data_limit)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=1, shuffle=True)
    
    train_dataset = vvad.Dataset_Frames("train", frames=syncnet_T, data_point_limit=data_limit)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=1, shuffle=True)
    
    if facial:
        print("Using Facial Edges")
        edges = st.facial_edges()
    else:
        print("Using Knn Edges")
        first_point = test_dataset[0]
        (x, x_rot, y) = first_point
        first_lmks = x[0].T
        edges = st.knn_edges(first_lmks)
    
    A = build_adjacency(V, edges)
    st_gcn_model = st.load_stgcn(st_gcn_checkpoint_dir, A, V, use_cuda, rotation=False) # Will need to adjust if I load from a checkpoint
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
    
    

    # eval_model(test_data_loader, device, vvad_model)
    
    # with torch.no_grad():
    #     prog_bar = enumerate(test_data_loader)
    #     for step, (x, x_rot, y) in prog_bar:
    #         x = x.permute(0, 2, 1, 3).to(device)

    #         # st_gcn_model(x)
    #         logits = vvad_model(x)
    #         preds = (logits > 0)
    #         print(preds)
    #         print(y)
    #         print()
            