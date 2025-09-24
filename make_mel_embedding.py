import os
import torch
import numpy as np

from hparams import hparams
import landmarks_audio as audio
from models.lmks_only import lmks_only
from models.audio_only import audio_only
import lmks_audio_eval as lmks_audio_eval

#################
# Audio generation functions
#################
####### For Color ########
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

def crop_audio_window(spec, num_frames=5, start_frame_num=0, video_fps=hparams.fps, mel_fps=80):
        mel_frames = int(num_frames * mel_fps / video_fps)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        end_idx = start_idx + mel_frames
        return spec[start_idx : end_idx, :]

####### For Landmarks #######
## Babble noise
babble_noise = "independent_scripts/babble_noise.wav" # '/home/ksw38/groups/grp_lip/nobackup/archive/datasets/speech-commands/_background_noise_/babble_noise.wav'
babble_wave = audio.load_wav(babble_noise, hparams.sample_rate)
babble_mel_global = audio.melspectrogram(babble_wave).T  # [Time, Mel]
def generate_babble_mel(num_frames=5, start_frame_num=0, video_fps=hparams.fps, mel_fps=80):
    babble_mel = crop_audio_window(babble_mel_global.copy(), num_frames=num_frames, start_frame_num=start_frame_num, video_fps=video_fps, mel_fps=mel_fps)  # Crop to the first mel step
    babble_mel = torch.FloatTensor(babble_mel.T).unsqueeze(0)  # [1, Mel, Time]
    return babble_mel

def generate_mel_from_path(wav_path, num_frames=5, start_frame_num=0, video_fps=hparams.fps, mel_fps=80):
    path_wave = audio.load_wav(wav_path, hparams.sample_rate)
    path_mel = audio.melspectrogram(path_wave).T
    path_mel = crop_audio_window(path_mel.copy(), num_frames=num_frames, start_frame_num=start_frame_num, video_fps=video_fps, mel_fps=mel_fps)
    path_mel = torch.FloatTensor(path_mel.T).unsqueeze(0)
    return path_mel

## Silence and White noise
def generate_mel_for_frames(num_frames, silence = True, video_fps=hparams.fps, mel_fps=80, sample_rate=16000, hop_length=200):
    mel_frames = int(num_frames * mel_fps / video_fps)
    num_samples = (mel_frames - 1) * hop_length  # +1 mel frame per hop
    if silence:
        gen_audio = torch.zeros(num_samples)
    else:
        gen_audio = torch.randn(num_samples) # Generate white noise
    # Compute mel spectrogram
    mel = audio.melspectrogram(gen_audio).T  # [Time, Mel]
    mel = mel[:mel_frames]  # Clip to exact mel_frames
    mel = torch.FloatTensor(mel.T).unsqueeze(0)  # [1, 80, mel_frames]
    return mel

################
# Model loading
################

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
        print("Missing keys in the state_dict:", missing)
    if unexpected:
        print("Unexpected keys in the state_dict:", unexpected)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    return model

def save_emb(mel, audio_model, output_path):
    mel = mel.to(device)
    print(mel.shape)
    emb = audio_model(mel)  # Get the embedding
    emb = emb.cpu().detach().numpy()  # Convert to numpy array
    # np.save(output_path, emb)
    # print(f"Embedding saved to {output_path}")

###############
# Main execution
###############
if __name__ == "__main__":
    device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = "triplets_checkpoints"
    checkpoint_path = None
    if checkpoint_path  is None:
            checkpoint_path = os.listdir(checkpoint_dir)[-1]
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

    babble_mel = generate_babble_mel(5, start_frame_num=0).to(torch.float32).unsqueeze(0)  # Generate a mel for 5 frames
    silent_mel = generate_mel_for_frames(5, silence=True).to(torch.float32).unsqueeze(0)

    audio_model = load_partial_model(checkpoint_path, device=device, startswith='audio')
    audio_model.eval()

    save_emb(babble_mel, audio_model, os.path.join("kimi", "babble_embedding.npy"))
    save_emb(silent_mel, audio_model, os.path.join("kimi", "silent_embedding.npy"))

    silence = torch.zeros(16000)
    print(f"Silence shape: {silence.shape}")
    silent_mel_color = lmks_audio_eval.cropped_mel(silence, start_frame_num=0)
    print(f"Cropped silent mel color shape: {silent_mel_color.shape}")
    batch_size = 1
    silent_mel_color = silent_mel_color.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    print(f"Silent mel color shape: {silent_mel_color.shape}")
    np.save(os.path.join("kimi", "silent_mel_color.npy"), silent_mel_color.cpu().numpy())