## LandmarkSTGCNConformer
print("Loading LandmarkSTGCNConformer Model from checkpoint {st_gcn_norot_checkpoint}")
st_gcn_model_norot = LandmarkSTGCNConformer(
    num_nodes=V,
    A=A,
    d_model=128,
    post_linear_hidden=128,
    conformer_layers=4,
    conformer_heads=4,
    conformer_ff=256,
    conformer_conv_kernel=31
)
st_gcn_model_norot.to(device)
st_gcn_norot_optimizer = optim.Adam([p for p in st_gcn_model_norot.parameters() if p.requires_grad],
                        lr=hparams.syncnet_lr, weight_decay=1e-5)
print('total trainable params for stgcn: {}'.format(sum(p.numel() for p in st_gcn_model_norot.parameters() if p.requires_grad)))
st_gcn_model_norot = load_from_checkpoint_or_dir(st_gcn_norot_checkpoint, model=st_gcn_model_norot, optimizer=st_gcn_norot_optimizer, use_cuda=use_cuda)
st_gcn_model_norot.eval()

print(f"\t and audio model from {audio_norot_checkpoint}")
audio_model_norot = audio_only().to(device)
optimizer = optim.Adam([p for p in audio_model_norot.parameters() if p.requires_grad],
                            lr=hparams.syncnet_lr, weight_decay=1e-5)
audio_model_norot = load_from_checkpoint_or_dir(audio_norot_checkpoint, model=audio_model_norot, optimizer=optimizer, use_cuda=use_cuda)
audio_model_norot.eval()

print(f"Loading LandmarkSTGCNConformer Model knn from checkpoint {st_gcn_norot_checkpoint_knn}")
st_gcn_model_norot_knn = LandmarkSTGCNConformer(
    num_nodes=V,
    A=A_knn,
    d_model=128,
    post_linear_hidden=128,
    conformer_layers=4,
    conformer_heads=4,
    conformer_ff=256,
    conformer_conv_kernel=31
)
st_gcn_model_norot_knn.to(device)
st_gcn_norot_knn_optimizer = optim.Adam([p for p in st_gcn_model_norot_knn.parameters() if p.requires_grad],
                        lr=hparams.syncnet_lr, weight_decay=1e-5)
print('total trainable params for stgcn: {}'.format(sum(p.numel() for p in st_gcn_model_norot_knn.parameters() if p.requires_grad)))
st_gcn_model_norot_knn = load_from_checkpoint_or_dir(st_gcn_norot_checkpoint_knn, model=st_gcn_model_norot_knn, optimizer=st_gcn_norot_knn_optimizer, use_cuda=use_cuda)
st_gcn_model_norot_knn.eval()

print(f"\t and audio model from {audio_norot_checkpoint_knn}")
audio_model_norot_knn = audio_only().to(device)
optimizer = optim.Adam([p for p in audio_model_norot_knn.parameters() if p.requires_grad],
                            lr=hparams.syncnet_lr, weight_decay=1e-5)
audio_model_norot_knn = load_from_checkpoint_or_dir(audio_norot_checkpoint_knn, model=audio_model_norot_knn, optimizer=optimizer, use_cuda=use_cuda)
audio_model_norot_knn.eval()

## LandmarkSTGCNConformerWithOrientation
print(f"Loading LandmarkSTGCNConformerWithOrientation Model from checkpoint {st_gcn_rot_checkpoint}")
st_gcn_model_rot = LandmarkSTGCNConformerWithOrientation(
    num_nodes=V,
    A=A,                # [K, V, V] adjacency
    d_model=128,
    post_linear_hidden=128,  # hidden size before conformer
    conformer_layers=4,
    conformer_heads=4,
    conformer_ff=256,
    conformer_conv_kernel=31
)
st_gcn_model_rot.to(device)
st_gcn_rot_optimizer = optim.Adam([p for p in st_gcn_model_rot.parameters() if p.requires_grad],
                        lr=hparams.syncnet_lr, weight_decay=1e-5)
print('total trainable params for stgcn: {}'.format(sum(p.numel() for p in st_gcn_model_rot.parameters() if p.requires_grad)))
st_gcn_model_norot_knn = load_from_checkpoint_or_dir(st_gcn_rot_checkpoint, model=st_gcn_model_rot, optimizer=st_gcn_rot_optimizer, use_cuda=use_cuda)
print('total trainable params for stgcn with rotation: {}'.format(sum(p.numel() for p in st_gcn_model_rot.parameters() if p.requires_grad)))
st_gcn_model_rot.eval()

print(f"\t and audio model from {audio_rot_checkpoint}")
audio_model_rot = audio_only().to(device)
optimizer = optim.Adam([p for p in audio_model_rot.parameters() if p.requires_grad],
                            lr=hparams.syncnet_lr, weight_decay=1e-5)
audio_model_rot = load_from_checkpoint_or_dir(audio_rot_checkpoint, model=audio_model_rot, optimizer=optimizer, use_cuda=use_cuda)
audio_model_rot.eval()