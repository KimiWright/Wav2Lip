import color_syncnet_train as train
from models import SyncNet_color as SyncNet
from hparams import hparams
import torch
from torch import nn
import torch.optim as optim
from torch.utils import data as data_utils
import os
import argparse

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    d = (d + 1) / 2 # Normalize to [0, 1]
    loss = logloss(d.unsqueeze(1), y)

    return loss

def eval_model_syncnet_task(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 20 ## Modification ##
    check_in_steps = 100
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):

            model.eval()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            losses.append(loss.item())

            if eval_steps is not None and step > eval_steps: break ## Modification ##
            if check_in_steps is not None and step % check_in_steps == 0: 
                averaged_loss = sum(losses) / len(losses)
                print(f"Step {step} averaged_loss: {averaged_loss}")

        averaged_loss = sum(losses) / len(losses)
        print(f"Final: {averaged_loss}")

        return

if __name__ == "__main__":
    checkpoint_dir = train.args.checkpoint_dir
    checkpoint_path = train.args.checkpoint_path
    checkpoint_path = "/home/ksw38/RVL/color_syncnet/Wav2Lip/lipsync_expert.pth"

    print("Loading checkpoint path")
    if checkpoint_path  is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    device = torch.device("cuda" if train.use_cuda else "cpu")
    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    print(f"Loading checkpoint from: {checkpoint_path}")
    if checkpoint_path is not None:
        train.load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    test_dataset = train.Dataset('val')

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=1)
    
    global_step = 0 # Placeholder for global step

    eval_model_syncnet_task(test_data_loader, global_step, device, model, checkpoint_dir)