"""Song features: YAMNet embeddings + 5 Librosa rhythm/loudness numbers (2,053 total).

Must stay identical to `extract_features` in notebooks/train_models_colab.ipynb,
because the trained models only understand features computed this way.
"""
import os

import librosa
import numpy as np
import onnxruntime as ort

from src.config import CLIP_SAMPLES, SAMPLE_RATE

EXTRA_NAMES = ["tempo", "rms_mean", "rms_std", "onset_mean", "pulse_clarity"]
FEATURE_NAMES = [f"emb_mean_{i}" for i in range(1024)] + [f"emb_std_{i}" for i in range(1024)] + EXTRA_NAMES


def available_cpus():
    """Cores this process may use (the container's limit on Linux), not the host's total."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return os.cpu_count() or 1


class FeatureExtractor:
    def __init__(self, yamnet_path):
        opts = ort.SessionOptions()
        # Explicit thread count: in a container onnxruntime sees all host cores and fails to pin threads to them.
        opts.intra_op_num_threads = available_cpus()
        opts.inter_op_num_threads = 1
        self.yamnet = ort.InferenceSession(str(yamnet_path), sess_options=opts, providers=["CPUExecutionProvider"])

    def __call__(self, y):
        """y: mono float32 waveform at 16 kHz, values in [-1, 1]. Returns 2,053 numbers."""
        y = np.asarray(y, dtype=np.float32)[:CLIP_SAMPLES]
        _, emb, _ = self.yamnet.run(None, {"waveform": y})
        onset = librosa.onset.onset_strength(y=y, sr=SAMPLE_RATE)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=SAMPLE_RATE)
        tempo = float(np.atleast_1d(tempo)[0])
        pulse = float(onset[beats].mean() / (onset.mean() + 1e-8)) if len(beats) else 0.0
        rms = librosa.feature.rms(y=y)[0]
        extras = np.array([tempo, rms.mean(), rms.std(), onset.mean(), pulse], dtype=np.float32)
        return np.nan_to_num(np.concatenate([emb.mean(0), emb.std(0), extras]).astype(np.float32))
