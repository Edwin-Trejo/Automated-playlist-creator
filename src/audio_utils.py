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

        # Cut or pad to exactly 30 s (22 050 × 30)
        target_length = 22050 * 30
        samples = np.pad(samples, (0, max(0, target_length - len(samples))))[:target_length]

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


def audio_to_mel(y, sr, n_mels=128, fixed_frames=640):
    """Convert audio to Mel-spectrogram identical to training preprocessing."""
    if y is None or sr is None:
        return None

    # Resample to 22,050 Hz (if needed)
    if sr != 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr = 22050

    # Generate mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Pad or trim to exactly 640 frames
    if mel_db.shape[1] < fixed_frames:
        pad_width = fixed_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mel_db = mel_db[:, :fixed_frames]

    # Add channel + batch dimension → (1, 128, 640, 1)
    mel_db = np.expand_dims(mel_db, axis=(0, -1))

    return mel_db