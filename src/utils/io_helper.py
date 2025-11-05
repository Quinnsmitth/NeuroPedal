# src/utils/io_helpers.py
import os
from pathlib import Path
from pedalboard.io import AudioFile
import soundfile as sf
import librosa
import numpy as np

# -----------------------------
# PATH HELPERS
# -----------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn’t exist, and return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def list_audio_files(directory: str | Path, extensions=(".wav", ".mp3", ".flac")) -> list[Path]:
    """Return a list of audio files in a given directory."""
    directory = Path(directory)
    return [p for p in directory.glob("*") if p.suffix.lower() in extensions]

# -----------------------------
# AUDIO READ/WRITE HELPERS
# -----------------------------

def read_audio(filepath: str | Path):
    """Read an audio file and return (audio_array, samplerate)."""
    filepath = Path(filepath)
    with AudioFile(filepath) as f:
        audio = f.read(f.frames)
        sr = f.samplerate
    return audio, sr

def write_audio(filepath: str | Path, audio: np.ndarray, sr: int):
    """Write an audio array to a .wav file using AudioFile."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with AudioFile(filepath, "w", sr, audio.shape[0]) as f:
        f.write(audio)

# -----------------------------
# LIBROSA / FEATURE HELPERS
# -----------------------------

def extract_mfcc(filepath: str | Path, n_mfcc=13):
    """Load file with librosa and return mean MFCC feature vector."""
    y, sr = librosa.load(filepath, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

def batch_extract_features(folder: str | Path, label_func=None):
    """
    Extract MFCCs from all audio files in a folder.
    Optionally use label_func(filename) to extract labels.
    Returns list of (features, label)
    """
    results = []
    for file in list_audio_files(folder):
        features = extract_mfcc(file)
        label = label_func(file.name) if label_func else None
        results.append((features, label))
    return results
