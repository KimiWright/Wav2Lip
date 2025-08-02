from run_statistics import *

class Dataset_5_Frame_Chunks_No_Short_Videos(object):
    def __init__(self, split = 'test'):
        global data_out_test, data_out_train
        data_point_limit = None
        if split == 'test':
            self.data = data_out_test
            if len(self.data) == 0:
                data_out_test = get_data(data_root, ground_truth, data_point_limit=data_point_limit)
                self.data = data_out_test
        elif split == 'train':
            self.data = data_out_train
            if len(self.data) == 0:
                data_out_train = get_data(train_data_root, ground_truth_train, data_point_limit=data_point_limit)
                self.data = data_out_train
        else:
            raise ValueError("Split must be 'test' or 'train'")
        
        self.processed_data = []
        modal_frame_num = 35 # Mode of frames in the dataset
        for datum in self.data:
            x_video, y = datum
            x_video_chunks = []
            num_frames = x_video.shape[0]
            if num_frames > modal_frame_num:
                for start_id in range(0, num_frames, 5):  # Get chunks of 5 frames
                    chunk = get_window_npy(x_video, start_id=start_id)
                    if chunk is not None:
                        x_video_chunks.append(chunk)
                x_video_chunks = np.array(x_video_chunks)
                if len(x_video_chunks) > 0:
                    self.processed_data.append((x_video_chunks, y))
                else:
                    print(f"Skipping data point {x_video.shape} due to insufficient frames for chunks")
    def __len__(self):
        return len(self.processed_data)
    def __getitem__(self, idx):
        return self.processed_data[idx]

babble_mel = generate_babble_mel(5, start_frame_num=0).to(torch.float32).unsqueeze(0)  # Generate a mel for 5 frames
def babble_loop_5_frame_chunk(model, data_loader, device): # Try the loss from contrastive learning
    with torch.no_grad():
        av_val_lists = []
        y_vals = []
        for step, (x, y) in enumerate(data_loader):
            x = x.to(device).to(torch.float32)
            # Change to [Chunk, Batch, Frames, Features] for easier iteration
            x = x.permute(1, 0, 2, 3)  # [Chunk, Batch, Frames, Features]
            y = y.to(device).to(torch.float32).unsqueeze(0)
            y_vals.append(int(y.item()))

            mel = babble_mel.to(device)

            av_vals = []
            for j in range(x.shape[0]):  # Iterate over the chunks
                try:
                    a, v = model(mel, x[j])  # x[j] is now [Batch, Frames, Features]
                    av_vals.append((a, v))
                except Exception as e:
                    print(f"Error processing chunk {j} in step {step}: {e}")
                    continue
            av_val_lists.append(av_vals)
        return y_vals, av_val_lists
    
def chunk_losses_single(y_vals, av_vals, device):
    losses = []
    for j, video in enumerate(av_vals):  # Iterate over videos
        a_vals, v_vals = zip(*video)  # Unzip a and v values
        a_vals = torch.stack(a_vals, dim=0)  # [Chunks, Batch, 128]
        v_vals = torch.stack(v_vals, dim=0)  # [Chunks, Batch, 128]
        a_mean = a_vals.mean(dim=0)  # Average over Chunks
        v_mean = v_vals.mean(dim=0)  # Average over Batch
        y = torch.Tensor([y_vals[j]]).unsqueeze(0)
        a_mean = a_mean.to(device)
        v_mean = v_mean.to(device)
        y = y.to(device)
        # loss = gru2.cosine_loss(a_mean, v_mean, y)
        loss = F.cosine_similarity(a_mean, v_mean)  # Cosine similarity loss
        losses.append(loss.cpu().item())  # Store loss on CPU to avoid GPU memory issues
    return losses


if __name__ == "__main__":
    device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shuffle_dataset = False
    num_workers = 1
    batch_size = 1
    threshold = 0.72 # Threshold for accuracy
    threshold = 0.1

    if checkpoint_path  is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                        lr=hparams.syncnet_lr)

    print("Loading checkpoint path")
    if checkpoint_path  is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
    load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)
    print("Loaded checkpoint from: {}".format(checkpoint_path))
    model.eval()

    test_dataset_5_frame_chunks = Dataset_5_Frame_Chunks_No_Short_Videos('test')
    print(f"\nNumber of samples in 5-frame chunks dataset: {len(test_dataset_5_frame_chunks)}")
    
    test_data_loader_5_frame_chunks = data_utils.DataLoader(
        test_dataset_5_frame_chunks, batch_size=batch_size,
        num_workers=8, drop_last=True, shuffle=shuffle_dataset)

    print("\nFinding AV vals for 5-frame chunks dataset")
    y_vals_5_frame_chunks, av_val_list = babble_loop_5_frame_chunk(model, test_data_loader_5_frame_chunks, device)

    # losses_5_frame_chunks = chunk_losses_single(y_vals_5_frame_chunks, av_val_list, device)    
    # acc_test = test_accuracy(losses_5_frame_chunks, y_vals_5_frame_chunks, threshold, flip=False)
    # print(f"Test accuracy: {acc_test}")

    num_chunks = 7 # Hardcoded number of chunks, can be adjusted based on the dataset, based on dataset with >35 frames
    all_losses = [[] for _ in range(num_chunks)]  # Initialize losses for each chunk
    for j, video in enumerate(av_val_list):  # Iterate over videos
        a_vals, v_vals = zip(*video)  # Unzip a and v values
        y = torch.Tensor([y_vals_5_frame_chunks[j]]).unsqueeze(0).to(device)
        vid_losses = []
        for k in range(num_chunks):
            a_val = a_vals[k].to(device)  # [Batch, 128]
            v_val = v_vals[k].to(device)  # [Batch, 128]
            # loss = gru2.cosine_loss(a_val, v_val, y)
            loss = F.cosine_similarity(a_val, v_val)
            vid_losses.append(loss.cpu().item())  # Store loss on CPU to avoid GPU memory issues
            all_losses[k].append(loss.cpu().item())  # Append loss to the corresponding

    chunk_accuracies = []
    for i, chunk_losses in enumerate(all_losses):
        acc = test_accuracy(chunk_losses, y_vals_5_frame_chunks, threshold, flip=True)
        print(f"Chunk {i} accuracy: {acc}")
        chunk_accuracies.append(acc * 100)  # Convert to percentage

    import matplotlib.pyplot as plt
    starting_frames = [i * 5 for i in range(num_chunks)]  # Starting frames for each chunk
    plt.figure()
    plt.plot(starting_frames, chunk_accuracies, marker='o', linestyle='-')
    plt.grid(True, which='both', axis='both')
    plt.ylim(0, 100)
    plt.xlim(0, max(starting_frames))
    plt.xlabel('Starting Frame', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(starting_frames)
    plt.savefig('chunk_accuracies.png')
    plt.close()