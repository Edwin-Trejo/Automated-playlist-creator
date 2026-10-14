import librosa
import numpy as np

from src.config import SAMPLE_RATE

SUMMARY_FEATURE_NAMES = (
    [f"mfcc_mean_{i}" for i in range(20)]
    + [f"mfcc_std_{i}" for i in range(20)]
    + [f"chroma_{i}" for i in range(12)]
    + ["centroid_mean", "centroid_std", "bandwidth_mean", "rolloff_mean"]
    + [f"contrast_{i}" for i in range(7)]
    + ["zcr_mean", "zcr_std", "rms_mean", "rms_std", "onset_mean", "onset_std",
       "tempo", "pulse_clarity", "percussive_ratio"]
)


def mel_spectrogram(y, sr=SAMPLE_RATE, n_mels=128, fixed_frames=640):
    """Mel-dB spectrogram identical to the CNN's training preprocessing."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < fixed_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, fixed_frames - mel_db.shape[1])), mode="constant")
    return mel_db[:, :fixed_frames].astype(np.float32)


def summary_vector(y, sr=SAMPLE_RATE):
    """Fixed-length audio descriptor used by the LightGBM models."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, tuning=0.0)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    pulse = float(onset_env[beats].mean() / (onset_env.mean() + 1e-8)) if len(beats) else 0.0
    _, percussive = librosa.effects.hpss(y)
    perc_ratio = float(np.sum(percussive ** 2) / (np.sum(y ** 2) + 1e-8))

    parts = [
        mfcc.mean(axis=1), mfcc.std(axis=1), chroma.mean(axis=1),
        [centroid.mean(), centroid.std(), bandwidth.mean(), rolloff.mean()],
        contrast.mean(axis=1),
        [zcr.mean(), zcr.std(), rms.mean(), rms.std(), onset_env.mean(), onset_env.std(),
         tempo, pulse, perc_ratio],
    ]
    vec = np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in parts])
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
