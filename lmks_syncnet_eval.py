from lmks_audio_eval import *
from landmarks_syncnet_eval import *

device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_dir = "triplets_checkpoints"

if checkpoint_path is None:
    checkpoint_path = os.listdir(checkpoint_dir)[-1]
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

model = load_face_model(checkpoint_path, device)
print('checkpoint path: {}'.format(checkpoint_path))
print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

test_dataset = Dataset(args.data_root, args.ground_truth)
test_data_loader = data_utils.DataLoader(
    test_dataset, batch_size=1,
    num_workers=8)


fconfms = []
losses = []
ys = []
cos_sim_vals = []
for step, (x_video, x_still, y) in enumerate(test_data_loader):
    x_video = x_video.to(device).to(torch.float32)
    x_still = x_still.to(device).to(torch.float32)
    y = y.to(device)
    
    feat_video = model(x_video)
    feat_still = model(x_still)

    fconfm = computeDist(feat_video, feat_still, vshift=15)
    fconfms.append(fconfm.item())
    ys.append(int(y.item()))

    # loss = cosine_loss(feat_video, feat_still, y.unsqueeze(0).to(torch.float32))
    # loss = cosine_loss(feat_video, feat_still, torch.ones((1, 1)).to(device))
    loss = cosine_loss(feat_video, feat_still, torch.zeros((1, 1)).to(device))
    losses.append(loss.item())

    cos_sim_val = nn.functional.cosine_similarity(feat_video, feat_still)
    cos_sim_vals.append(cos_sim_val.item())

num_ys_0 = ys.count(0)
num_ys_1 = ys.count(1)
print(num_ys_0, num_ys_1)
print("fconfm")
best_accuracy(fconfms, ys)
best_accuracy(fconfms, ys, flip=True)

print("cosine loss")
best_accuracy(losses, ys)
best_accuracy(losses, ys, flip=True)

print("cosine similarity")
best_accuracy(cos_sim_vals, ys)
best_accuracy(cos_sim_vals, ys, flip=True)