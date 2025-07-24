import os
import torch
import numpy as np

from hparams import hparams
import landmarks_audio as audio
from models.lmks_only import lmks_only
from models.audio_only import audio_only

#################
# Audio generation functions
#################

def crop_audio_window(spec, num_frames=5, start_frame_num=0, video_fps=hparams.fps, mel_fps=80):
        mel_frames = int(num_frames * mel_fps / video_fps)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        end_idx = start_idx + mel_frames
        return spec[start_idx : end_idx, :]

## Babble noise
babble_noise = '/home/ksw38/groups/grp_lip/nobackup/archive/datasets/speech-commands/_background_noise_/babble_noise.wav'
babble_wave = audio.load_wav(babble_noise, hparams.sample_rate)
babble_mel_global = audio.melspectrogram(babble_wave).T  # [Time, Mel]
def generate_babble_mel(num_frames=5, start_frame_num=0, video_fps=hparams.fps, mel_fps=80):
    babble_mel = crop_audio_window(babble_mel_global.copy(), num_frames=num_frames, start_frame_num=start_frame_num, video_fps=video_fps, mel_fps=mel_fps)  # Crop to the first mel step
    babble_mel = torch.FloatTensor(babble_mel.T).unsqueeze(0)  # [1, Mel, Time]
    return babble_mel

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


###############
# Main execution
###############
device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_dir = "triplets_checkpoints"
checkpoint_path = None
if checkpoint_path  is None:
        checkpoint_path = os.listdir(checkpoint_dir)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

babble_mel = generate_babble_mel(5, start_frame_num=0).to(torch.float32).unsqueeze(0)  # Generate a mel for 5 frames

audio_model = load_partial_model(checkpoint_path, device=device, startswith='audio')
audio_model.eval()

babble_mel = babble_mel.to(device)

babble_emb = audio_model(babble_mel)  # Get the babble embedding
babble_emb = babble_emb.cpu().detach().numpy()  # Convert to numpy array
print(f"Babble embedding shape: {babble_emb.shape}")

output_path = os.path.join("kimi", "babble_embedding.npy")
np.save(output_path, babble_emb)
print(f"Babble embedding saved to {output_path}")