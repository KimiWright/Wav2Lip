from color_syncnet_train import *

if __name__ == "__main__":
    checkpoint_dir = "checkpoints"
    checkpoint_path = None

    print("Loading checkpoint path")
    if checkpoint_path  is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    print(f"Loading checkpoint from: {checkpoint_path}")
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    test_dataset = Dataset('val')

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=1)
    
    global_step = 0 # Placeholder for global step

    eval_model(test_data_loader, global_step, device, model, checkpoint_dir)