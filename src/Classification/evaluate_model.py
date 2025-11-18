# src/Classification/evaluate_model.py

import sys
from pathlib import Path
import torch
from tqdm import tqdm
from pedalboard.io import AudioFile

# Ensure project root + src is in sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from select_path import load_config
from melSpec import mel_spectrogram
from model import PedalResNet


def load_wav_as_mel(path: Path, target_length: int = 160000):
    """Load audio -> mono -> pad/trim -> mel spectrogram."""
    with AudioFile(str(path)) as f:
        audio = f.read(f.frames)
        sr = f.samplerate

    audio = torch.tensor(audio, dtype=torch.float32)

    if audio.ndim == 2:
        audio = audio.mean(dim=0, keepdim=True)

    # enforce consistent length
    if audio.size(1) > target_length:
        audio = audio[:, :target_length]
    elif audio.size(1) < target_length:
        pad = target_length - audio.size(1)
        audio = torch.nn.functional.pad(audio, (0, pad))

    mel = mel_spectrogram(audio)      # (1, mels, time)
    mel = mel.unsqueeze(0)            # (1, 1, mels, time)
    return mel


def parse_drive_tone(filename: str):
    """Extract drive/tone from filename like: riff_drive40_tone60.wav"""
    parts = filename.split("_")
    drive = int([p for p in parts if p.startswith("drive")][0].replace("drive", ""))
    tone = int([p for p in parts if p.startswith("tone")][0].replace("tone", ""))
    return drive, tone


def evaluate_model(weights_path: Path, distorted_dir: Path):
    print(f"Loading model: {weights_path}")

    model = PedalResNet(output_size=2, use_pretrained=False)
    model.load_weights(weights_path)
    model.eval()

    wav_files = list(distorted_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {distorted_dir}")

    total_drive_error = 0
    total_tone_error = 0
    count = 0

    print(f"\nEvaluating {len(wav_files)} files...\n")

    for f in tqdm(wav_files, desc="Evaluating"):
        mel = load_wav_as_mel(f)
        gt_drive, gt_tone = parse_drive_tone(f.stem)

        with torch.no_grad():
            pred = model(mel)[0] * 10     # undo normalization
            drive_pred, tone_pred = pred.tolist()

        total_drive_error += abs(drive_pred - gt_drive)
        total_tone_error += abs(tone_pred - gt_tone)
        count += 1

    mean_drive_error = total_drive_error / count
    mean_tone_error = total_tone_error / count
    mean_total_error = (mean_drive_error + mean_tone_error) / 2

    print("\nEvaluation Results")
    print("----------------------------------")
    print(f"Mean Drive Error: {mean_drive_error:.3f}")
    print(f"Mean Tone Error : {mean_tone_error:.3f}")
    print(f"Average Error   : {mean_total_error:.3f}")
    print("----------------------------------")

    return mean_drive_error, mean_tone_error, mean_total_error


if __name__ == "__main__":
    root = load_config()  # USB root
    distorted_dir = root / "distorted"

    project_root = Path(__file__).resolve().parents[2]
    weights_path = project_root / "weights" / "guitar_model_mel.pth"

    evaluate_model(weights_path, distorted_dir)
