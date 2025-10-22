import requests
import librosa
import numpy as np
import io
from pydub import AudioSegment

def download_audio(preview_url, duration=30):
    """Download Deezer preview MP3 and decode it to waveform using pydub."""
    try:
        import requests
        response = requests.get(preview_url, timeout=10)
        response.raise_for_status()

        # Convert MP3 bytes → AudioSegment (via pydub + ffmpeg)
        audio_bytes = io.BytesIO(response.content)
        audio = AudioSegment.from_file(audio_bytes, format="mp3")

        # Convert to numpy array for librosa
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if audio.channels == 2:  # stereo → mono
            samples = samples.reshape((-1, 2)).mean(axis=1)

        # Normalize between -1 and 1
        samples /= np.max(np.abs(samples))

        # librosa expects sample rate; pydub gives it as audio.frame_rate
        y = samples
        sr = audio.frame_rate

        # Optionally trim to desired duration (seconds)
        max_samples = int(sr * duration)
        y = y[:max_samples]

        return y, sr
    except Exception as e:
        print(f"⚠️  Error loading audio: {e}")
        return None, None


def audio_to_mel(y, sr, n_mels=128, n_fft=2048, hop_length=512):
    """Convert waveform to normalized 128×128 Mel spectrogram."""
    if y is None or sr is None:
        return None

    # Generate mel spectrogram (power)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )

    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Resize to 128×128 for CNN input
    mel_spec_resized = librosa.util.fix_length(mel_spec_db, size=128, axis=1)

    # Normalize between 0 and 1
    mel_norm = (mel_spec_resized - mel_spec_resized.min()) / (
        mel_spec_resized.max() - mel_spec_resized.min()
    )

    # Expand dimensions → shape (1, 128, 128, 1)
    mel_norm = np.expand_dims(mel_norm, axis=(0, -1))

    return mel_norm
