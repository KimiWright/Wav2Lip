from lmks_syncnet_train_gru2_contrastive_loss import *
import os
import torch
import torch.optim as optim
import numpy as np

def is_normalized(vectors, tolerance=1e-6):
    """
    Check if each vector in a list or array is normalized.
    
    Parameters:
        vectors (list or np.ndarray): A list or array of vectors.
        tolerance (float): Acceptable deviation from 1 for the norm.
        
    Returns:
        list of bool: True if the corresponding vector is normalized, else False.
    """
    vectors = np.array(vectors)
    norms = np.linalg.norm(vectors, axis=1)
    return np.abs(norms - 1.0) < tolerance

checkpoint_dir = args.checkpoint_dir
checkpoint_path = args.checkpoint_path

if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

# Dataset and Dataloader setup
test_dataset = Dataset('val')

batch_size = 1 #hparams.syncnet_batch_size
test_data_loader = data_utils.DataLoader(
    test_dataset, batch_size=batch_size,
    num_workers=8)

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SyncNet().to(device)
print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                    lr=hparams.syncnet_lr, weight_decay=1e-5) # Try adding weight decay

reset_optimizer = False
print("Loading checkpoint path")
if checkpoint_path is not None:
    load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=reset_optimizer)
else:
    checkpoint_path = os.listdir(checkpoint_dir)[-1]
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
    load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=reset_optimizer)
print("Loaded checkpoint from: {}".format(checkpoint_path))

model.eval()

a_vals = []
v_vals = []
cos_sim_vals = []
y_vals = []
for step, (x, mel, y) in enumerate(test_data_loader):
    x = x.to(device)
    mel = mel.to(device)
    y = y.to(device)
    optimizer.zero_grad()
    
    a, v = model(mel, x)
    a_vals.append(a.detach().cpu().numpy())
    v_vals.append(v.detach().cpu().numpy())

    cos_sim = F.cosine_similarity(a, v)
    cos_sim_vals.append(cos_sim.detach().cpu().numpy())

    y_vals.append(y.detach().cpu().numpy())

    if step == 5:
        break

y_tensor = torch.tensor(y_vals).squeeze()
cos_sim_tensor = torch.tensor(cos_sim_vals).squeeze()
print(y_vals)
print(cos_sim_vals)

print(y_tensor.shape)
print(cos_sim_tensor.shape)

print("Cosine similarities (pos):", cos_sim_tensor[y_tensor == 1].mean().item())
print("Cosine similarities (neg):", cos_sim_tensor[y_tensor == 0].mean().item())

margin = 0.2
print("Negatives over margin:", (cos_sim_tensor[y_tensor == 0] > margin).float().mean().item())




# print("Audio Vectors Normalization Check:")
# for val in a_vals:
#     norm = is_normalized(val)
#     if not norm.all():
#         print("No")
#     else:
#         print("Yes")

# print("Visual Vectors Noralization Check:")
# for val in v_vals:
#     norm = is_normalized(val)
#     if not norm.all():
#         print("No")
#     else:
#         print("Yes")