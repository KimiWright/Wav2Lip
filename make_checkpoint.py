import torch
from torch import optim
import os
from os.path import join
from models import SyncNet_landmarks_gru2 as SyncNet
from hparams import hparams

new_checkpoint_dir = "pre_loaded_audio_checkpoints"
new_checkpoint_name = "audio_loaded.pth"

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, new_checkpoint_name)
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

if __name__ == "__main__":
    checkpoint_path = None
    checkpoint_dir = 'landmarks_checkpoints_gru2'
    if checkpoint_path is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    full_state_dict = checkpoint['state_dict']

    face_state_dict = {k: v for k, v in full_state_dict.items() if k.startswith("face")}

    color_syncnet_checkpoint = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints/checkpoint_step000510000.pth"
    color_syncnet = torch.load(color_syncnet_checkpoint, map_location='cpu')
    color_syncnet_state_dict = color_syncnet['state_dict']
   
    audio_state_dict = {k: v for k, v in color_syncnet_state_dict.items() if k.startswith("audio")}
    face_state_dict.update(audio_state_dict)

    leftover_keys = ['audio_proj.0.weight', 'audio_proj.0.bias', 'audio_proj.2.weight', 'audio_proj.2.bias']
    leftover_dict = {k: v for k, v in full_state_dict.items() if k in leftover_keys}
    face_state_dict.update(leftover_dict)
    
    model = SyncNet()
    missing, unexpected = model.load_state_dict(face_state_dict, strict=False)
    if missing:
        print("Missing keys in the state_dict:", missing)
    if unexpected:
        print("Unexpected keys in the state_dict:", unexpected)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                        lr=hparams.syncnet_lr)
    
    optimizer_state = checkpoint['optimizer']
    optimizer.load_state_dict(optimizer_state)
    step = checkpoint['global_step']
    epoch = checkpoint['global_epoch']

    save_checkpoint(model, optimizer, step, new_checkpoint_dir, epoch)
    print("Checkpoint saved successfully at", new_checkpoint_dir)
    print("as ", new_checkpoint_name)


