import os
import torch
import torch.optim as optim

from hparams import hparams
from models.lmks_only import lmks_only
from models.audio_only import audio_only
from models import SyncNet_landmarks_gru2 as SyncNet

######################
# Load Partial Models
######################

def load_partial_model(checkpoint_path, device, startswith='face'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    full_state_dict = checkpoint['state_dict']

    partial_state_dict = {k: v for k, v in full_state_dict.items() if k.startswith(startswith)}

    if startswith == 'face':
        model = lmks_only().to(device)
    elif startswith == 'audio':
        model = audio_only().to(device)
    else:
        raise ValueError("startswith must be 'face' or 'audio'")
    missing, unexpected = model.load_state_dict(partial_state_dict, strict=False)

    if missing:
        print("Trying to load with new keys...")
        new_state_dict = {}
        for k, v in partial_state_dict.items():
            new_key = k.split('.', 1)[1] if '.' in k else k
            new_state_dict[new_key] = v
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    if missing:
        print("Missing keys in the state_dict:", missing)
    if unexpected:
        print("Unexpected keys in the state_dict:", unexpected)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    return model

def print_non_matching(list1, list2):
    min_len = min(len(list1), len(list2))
    matching = True
    for i in range(min_len):
        if list1[i] != list2[i]:
            print(f"Mismatch at index {i}: {list1[i]} != {list2[i]}")
            matching = False

    # Print extra items in the longer list
    if len(list1) > len(list2):
        print(f"Extra items in list1: {list1[len(list2):]}")
    elif len(list2) > len(list1):
        print(f"Extra items in list2: {list2[len(list1):]}")
    else:
        print("Lists are the same length.")

    if matching:
        print("All items match up to the length of the shorter list.")

######################
# Load Full Model Checkpoint
######################

def _load(checkpoint_path, use_cuda=False):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False, use_cuda=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path, use_cuda=use_cuda)
    full_state_dict = checkpoint["state_dict"]
    missing, unexpected = model.load_state_dict(full_state_dict, strict=False)

    if missing:
        print("Trying to load with new keys...")
        new_state_dict = {}
        for k, v in full_state_dict.items():
            new_key = k.split('.', 1)[1] if '.' in k else k
            new_state_dict[new_key] = v
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print("Missing keys in the state_dict:", missing)
    if unexpected:
        print("Unexpected keys in the state_dict:", unexpected)

    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model

device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
shuffle_dataset = False
num_workers = 1

checkpoint_dir_triplets = "triplets_checkpoints"
checkpoint_dir_finetune = "finetune_checkpoints"

checkpoint_path_triplets = os.listdir(checkpoint_dir_triplets)[-1]
checkpoint_path_triplets = os.path.join(checkpoint_dir_triplets, checkpoint_path_triplets)
checkpoint_path_finetune = os.listdir(checkpoint_dir_finetune)[-1]
checkpoint_path_finetune = os.path.join(checkpoint_dir_finetune, checkpoint_path_finetune)

#########################
# SyncNet Model
#########################

model = SyncNet().to(device)
print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                        lr=hparams.syncnet_lr, weight_decay=1e-5) # Try adding weight decay
load_checkpoint(checkpoint_path_finetune, model, optimizer, reset_optimizer=False, use_cuda=False)

#######################
# Partial Models
#######################

face_model = load_partial_model(checkpoint_path_finetune, device, startswith='face')
audio_model = load_partial_model(checkpoint_path_finetune, device, startswith='audio')

checkpoint = torch.load(checkpoint_path_triplets, map_location='cpu')
full_state_dict_triplets = checkpoint['state_dict']

# print(full_state_dict.keys())
# print()

checkpoint = torch.load(checkpoint_path_finetune, map_location='cpu')
full_state_dict_finetuned = checkpoint['state_dict']
after_period = [s.split('.', 1)[1] if '.' in s else '' for s in full_state_dict_finetuned.keys()]


list1 = list(full_state_dict_triplets.keys())
list2 = after_period

print_non_matching(list1, list2)
