import landmarks_audio as audio
import torch
import os, argparse
from torch import optim
from hparams import hparams
import landmarks_syncnet_train_gru2 as gru2
from models import SyncNet_landmarks_gru2 as SyncNet


parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default='landmarks_checkpoints_gru2', type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)

args = parser.parse_args()

checkpoint_dir = args.checkpoint_dir
checkpoint_path = args.checkpoint_path

syncnet_mel_step_size = 16

def crop_audio_window(spec, start_frame_num):
        
    start_idx = int(80. * (start_frame_num / float(hparams.fps)))

    end_idx = start_idx + syncnet_mel_step_size

    return spec[start_idx : end_idx, :]

def cropped_mel(audio_tensor, start_frame_num=0):
    mel = audio.melspectrogram(audio_tensor).T # shape: (Time, Mel)
    cropped_mel = crop_audio_window(mel.copy(), start_frame_num)
    mel = torch.FloatTensor(cropped_mel.T).unsqueeze(0)  # [1, Mel, Time]
    return mel


if __name__ == "__main__":
    # Generate 1 second of silence
    silence = torch.zeros(16000)  # 1 second at 16kHz
    white_noise = torch.randn(16000)

    ### Make a 5 frame Mel spectrogram and a full one for comparison ###
    # Starting with the 5 frame.

    test_dataset = gru2.Dataset('val')

    device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                        lr=hparams.syncnet_lr)

    print("Loading checkpoint path")
    if checkpoint_path  is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    gru2.load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)
    print("Loaded checkpoint from: {}".format(checkpoint_path))

    x, mel, y = test_dataset[0]
    model.eval()

    # Transform data to CUDA device
    x = x.to(device)

    mel = mel.to(device)

    print("Model")

    a, v = model(mel, x)
    y = y.to(device)
    print("Model run complete")
    loss = gru2.cosine_loss(a, v, y)

    silent_mel = cropped_mel(silence, start_frame_num=0).to(device)
    white_noise_mel = cropped_mel(white_noise, start_frame_num=0).to(device)
    print("Mel shape: {}".format(mel.shape))
    print("Silent mel shape: {}".format(silent_mel.shape))
    print("White noise mel shape: {}".format(white_noise_mel.shape))

    silent_a, silent_v = model(silent_mel, x)
    white_noise_a, white_noise_v = model(white_noise_mel, x)

    silent_loss = gru2.cosine_loss(silent_a, silent_v, y)
    white_noise_loss = gru2.cosine_loss(white_noise_a, white_noise_v, y)
    print("Loss on test data: {}".format(loss.item()))
    print("Loss on silent audio: {}".format(silent_loss.item()))
    print("Loss on white noise audio: {}".format(white_noise_loss.item()))
    