# src/Classification/infer.py
import sys
from pathlib import Path

# Ensure project root is in Python path
project_root = Path(__file__).resolve().parents[1]
print("PROJECT ROOT:", project_root)
sys.path.append(str(project_root))

import random
import torch
from pedalboard.io import AudioFile

from select_path import load_config
from melSpec import mel_spectrogram
from model import PedalResNet


def load_wav_as_mel(path: Path, target_length: int = 160000):
    """
    Loads WAV → mono → pad/trim → mel (DB).
    Matches the training pipeline exactly.
    """
    try:
        with AudioFile(str(path)) as f:
            audio = f.read(f.frames)
            sr = f.samplerate
    except Exception as e:
        raise RuntimeError(f"Could not read WAV: {path} ({e})")

    # Convert to tensor
    audio = torch.tensor(audio, dtype=torch.float32)

    # Stereo → mono
    if audio.ndim == 2:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Pad/trim to 4 seconds (40–41kHz)
    if audio.size(1) > target_length:
        audio = audio[:, :target_length]
    elif audio.size(1) < target_length:
        pad_amount = target_length - audio.size(1)
        audio = torch.nn.functional.pad(audio, (0, pad_amount))

    # Compute mel spectrogram
    mel_db = mel_spectrogram(audio)  # (1, 128, time)

    return mel_db.unsqueeze(0)  # → (1, 1, 128, time)


def list_valid_wavs(distorted_dir: Path):
    """
    Returns all valid WAV files, skipping:
    - macOS metadata files (._*)
    - tiny files (< 1kb)
    """
    wavs = [
        w for w in distorted_dir.glob("*.wav")
        if not w.name.startswith("._") and w.stat().st_size > 2000
    ]
    return wavs


def get_random_distorted_wav(root):
    """
    Picks a random .wav file from USB/distorted folder.
    Filters out invalid macOS metadata files automatically.
    """
    distorted_dir = root / "distorted"
    wavs = list_valid_wavs(distorted_dir)

    if not wavs:
        raise FileNotFoundError(
            f"No valid WAV files found in {distorted_dir}. "
            "Do you have only macOS ._ metadata files?"
        )

    wav_path = random.choice(wavs)
    print(f"\nSelected random WAV: {wav_path.name}")
    return wav_path


def predict_file(wav_path: Path, weights_path: Path):
    """
    Runs inference on a single WAV file.
    """
    print(f"\nLoading WAV: {wav_path}")

    mel = load_wav_as_mel(wav_path)  # (1, 1, 128, time)

    model = PedalResNet(output_size=2, use_pretrained=False)
    model.load_weights(weights_path)
    model.eval()

    with torch.no_grad():
        preds = model(mel)

    # Undo label scaling (y/10 in training)
    preds = preds * 10
    drive, tone = preds[0].tolist()

    # Round to nearest whole number
    drive = round(drive)
    tone = round(tone)

    print(f"\nPredicted Parameters:")
    print(f"   Drive: {drive}")
    print(f"   Tone : {tone}\n")

    return drive, tone



if __name__ == "__main__":
    # USB root (contains distorted/, clean/, etc)
    root = load_config()

    # Project root (where weights/ lives)
    project_root = Path(__file__).resolve().parents[2]
    weights_path = project_root / "weights" / "guitar_model_mel_36300_100.pth"

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing model weights at: {weights_path}")

    # Pick a random distorted WAV
    wav_path = get_random_distorted_wav(root)

    # Run prediction
    predict_file(wav_path, weights_path)

    print(f"\nUsed file: {wav_path}\n")
