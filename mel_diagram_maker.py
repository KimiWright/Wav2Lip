import hparams as hp
from landmarks_audio import melspectrogram
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


# ------------------------------------------------------------------
# 2. Load waveform
# ------------------------------------------------------------------
audio_path = r"C:\Users\Kimi\Documents\RVL\MultiAVSR\datamodule\babble_noise.wav"   # path to your recording

sr, wav = wavfile.read(audio_path)

# If stereo, convert to mono
if wav.ndim > 1:
    wav = wav.mean(axis=1)

# Convert to float32 in [-1, 1]
wav = wav.astype(np.float32)
max_val = np.max(np.abs(wav))
if max_val > 0:
    wav /= max_val


# ------------------------------------------------------------------
# 3. Compute mel spectrogram (your function)
#    Your function:
#    def melspectrogram(wav):
#        D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
#        S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db
#        if hp.signal_normalization:
#            return _normalize(S)
#        return S
# ------------------------------------------------------------------
mel = melspectrogram(wav)      # shape: [n_mels, T]


# ------------------------------------------------------------------
# 4. Plot and save image
# ------------------------------------------------------------------
plt.figure(figsize=(8, 4))
im = plt.imshow(
    mel,
    origin="lower",
    aspect="auto",
    interpolation="nearest"
)
plt.title("Mel spectrogram of 'lemon'")
plt.xlabel("Time frames")
plt.ylabel("Mel bin")
plt.colorbar(im, label="dB")
plt.tight_layout()
plt.savefig("mel_spectrogram_lemon.png", dpi=300)
plt.show()
