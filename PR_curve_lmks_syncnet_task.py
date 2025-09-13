import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils import data as data_utils
from hparams import hparams
import landmarks_syncnet_train_gru2 as train

### Set Variables ###
eval_step_max = None
model_types = ["gru3", "gru2", "attn", "triplets"]
model_type = model_types[-1]

if model_type == "gru3":
    from models import SyncNet_landmarks_gru3 as SyncNet
    checkpoint_dir = "checkpoints_gru3"
    checkpoint_path = None
if model_type == "gru2":
    from models import SyncNet_landmarks_gru2 as SyncNet
    checkpoint_dir = 'landmarks_checkpoints_gru2'
    checkpoint_path = None
if model_type == "attn":
    from models import SyncNet_landmarks_attn as SyncNet
    checkpoint_dir = 'attn_checkpoints'
    checkpoint_path = None
if model_type == "triplets":
    from models import SyncNet_landmarks_gru2 as SyncNet
    checkpoint_dir = "triplets_checkpoints"
    checkpoint_path = None
else:
    raise(ValueError(f"{model_type} is not a model type, choose from: {model_types}"))

fig_path = f"PR_curve_lmks_{model_type}.png"
fig_title = f"Precision-Recall for {model_type} on determining if audio and video are synced"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

### Eval Loop ###

def eval_model_syncnet_task(test_data_loader, device, model):
    eval_steps = eval_step_max
    check_in_steps = 100
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    y_truth = []

    for step, (x, mel, y) in enumerate(test_data_loader):
        model.eval()

        # Transform data to CUDA device
        x = x.to(device)

        mel = mel.to(device)

        a, v = model(mel, x)
        y = y.to(device)

        loss = F.cosine_similarity(a, v)

        losses.append(-loss.item()) ## Test ##
        y_truth.append(y.item())

        if eval_steps is not None and step > eval_steps: break ## Modification ##
        if check_in_steps is not None and step % check_in_steps == 0: 
            averaged_loss = sum(losses) / len(losses)
            print(f"Step {step} averaged_loss: {averaged_loss}")

    averaged_loss = sum(losses) / len(losses)
    print(f"Final: {averaged_loss}")

    return y_truth, losses

### Set Up ###
model = SyncNet().to(device)
print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                        lr=hparams.syncnet_lr)

print(f"Loading checkpoint from: {checkpoint_path}")
if checkpoint_path is not None:
    train.load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False, use_cuda=use_cuda)

test_dataset = train.Dataset('val')

test_data_loader = data_utils.DataLoader(
    test_dataset, batch_size=1,
    num_workers=1)

### Create PR Curve ###

y_test, y_scores = eval_model_syncnet_task(test_data_loader, device, model)

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
auc_score = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {auc_score:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(fig_title)
plt.legend()
plt.savefig(fig_path)