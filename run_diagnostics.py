from landmarks_syncnet_train_gru2 import *
import torch

def run_diagnostics(model, data_loader, device):
    model.train()
    x, mel, y = next(iter(data_loader))  # Single batch
    x, mel, y = x.to(device), mel.to(device), y.to(device)

    # Forward pass
    a, v = model(mel, x)
    loss = cosine_loss(a, v, y)
    print(f"Loss: {loss.item():.4f}")

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Check for NaNs or zero grads
    dead_layers = []
    grad_info = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                grad_info.append((name, "No grad"))
                dead_layers.append(name)
            elif torch.isnan(param.grad).any():
                grad_info.append((name, "NaN grad"))
                dead_layers.append(name)
            elif param.grad.abs().sum() == 0:
                grad_info.append((name, "Zero grad"))
                dead_layers.append(name)
            else:
                grad_info.append((name, f"Grad norm: {param.grad.norm().item():.4e}"))

    print("\n=== Gradient Diagnostic ===")
    for name, status in grad_info:
        print(f"{name}: {status}")

    # Optional: Check parameter updates
    with torch.no_grad():
        before = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                before[name] = param.clone()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.step()

        changed = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                delta = (param - before[name]).abs().sum().item()
                if delta > 0:
                    changed.append((name, delta))

    print("\n=== Parameter Updates After Optimizer Step ===")
    for name, delta in changed:
        print(f"{name}: changed by {delta:.4e}")
    
    if not changed:
        print("No parameters updated — check optimizer or frozen params.")

    if dead_layers:
        print("\n Warning: These layers may be dead or misconfigured:")
        for name in dead_layers:
            print(f"  • {name}")

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup
    test_dataset = Dataset('val')
    train_dataset = Dataset('train')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=8)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                        lr=hparams.syncnet_lr, weight_decay=1e-5) # Try adding weight decay

    # Learning rate scheduler
    total_steps = len(train_data_loader) * hparams.nepochs
    warmup_steps = int(total_steps * 0.1)  # 10% of total steps for warmup
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    reset_optimizer = False # For the next training session the current continues to stagnate. Current session was set to False
    print("Loading checkpoint path")
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=reset_optimizer)
    else:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=reset_optimizer)
    print("Loaded checkpoint from: {}".format(checkpoint_path))
    print()

    # Run diagnostics
    print("Running diagnostics...")
    run_diagnostics(model, train_data_loader, device)
    print("Diagnostics complete.")
