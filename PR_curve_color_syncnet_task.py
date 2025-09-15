import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
import torch.nn.functional as F
import torch
import color_syncnet_train as train
from models import SyncNet_color as SyncNet
from hparams import hparams
import torch.optim as optim
from torch.utils import data as data_utils

fig_path = "PR_curve_color_syncnet.png"
eval_step_max = 100
checkpoint_path = train.args.checkpoint_path

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


## Color Syncnet Model ##

# def eval_model_syncnet_task(test_data_loader, device, model):
#     eval_steps = eval_step_max
#     check_in_steps = 100
#     print('Evaluating for {} steps'.format(eval_steps))
#     losses = []
#     y_truth = []

#     for step, (x, mel, y) in enumerate(test_data_loader):
#         model.eval()

#         # Transform data to CUDA device
#         x = x.to(device)

#         mel = mel.to(device)

#         a, v = model(mel, x)
#         y = y.to(device)

#         print(y)

#         loss = F.cosine_similarity(a, v)

#         losses.append(loss.item())
#         y_truth.append(y.item())

#         if eval_steps is not None and step > eval_steps: break ## Modification ##
#         if check_in_steps is not None and step % check_in_steps == 0: 
#             averaged_loss = sum(losses) / len(losses)
#             print(f"Step {step} averaged_loss: {averaged_loss}")

    # averaged_loss = sum(losses) / len(losses)
    # print(f"Final: {averaged_loss}")

    # return y_truth, losses

def eval_model_syncnet_task(test_loader, device, model):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for step, (x, mel, y) in enumerate(test_loader):
            x = x.to(device, non_blocking=True)
            mel = mel.to(device, non_blocking=True)
            a, v = model(mel, x)                           # [B, D] embeddings
            s = F.cosine_similarity(a, v, dim=1)           # [B] higher = more synced

            # Collect all items in batch safely
            y_true.extend(y.view(-1).detach().cpu().tolist())
            y_score.extend(s.view(-1).detach().cpu().tolist())

            if eval_step_max is not None and step >= eval_step_max:
                break

model = SyncNet().to(device)
print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                        lr=hparams.syncnet_lr)

print(f"Loading checkpoint from: {checkpoint_path}")
if checkpoint_path is not None:
    train.load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False, use_cuda=use_cuda)
model.eval()


test_dataset = train.Dataset('val')

test_data_loader = data_utils.DataLoader(
    test_dataset, batch_size=1,
    num_workers=1)

y_test, y_scores = eval_model_syncnet_task(test_data_loader, device, model)

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
auc_score = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {auc_score:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig(fig_path)