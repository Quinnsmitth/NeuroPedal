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
import sys
from pathlib import Path
import re
import torch
import numpy as np
from pedalboard.io import AudioFile
from select_path import load_config
from melSpec import mel_spectrogram
from model import PedalResNet
from sklearn.metrics import confusion_matrix


# ----------------------------------------------------------
# Load WAV → mono → pad/trim → mel db
# ----------------------------------------------------------
def load_wav_as_mel(path: Path, target_length: int = 160000):
    try:
        with AudioFile(str(path)) as f:
            audio = f.read(f.frames)
            sr = f.samplerate
    except Exception as e:
        raise RuntimeError(f"Could not read WAV: {path} ({e})")

    audio = torch.tensor(audio, dtype=torch.float32)

    # stereo → mono
    if audio.ndim == 2:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # pad/trim
    if audio.size(1) > target_length:
        audio = audio[:, :target_length]
    else:
        audio = torch.nn.functional.pad(audio, (0, target_length - audio.size(1)))

    mel = mel_spectrogram(audio)
    return mel.unsqueeze(0)


# ----------------------------------------------------------
# Parse true parameters from filename
# e.g. clean_riff_000_drive80_tone60.wav
# ----------------------------------------------------------
def parse_filename(fname: str):
    pat = r"drive(\d+)_tone(\d+)"
    match = re.search(pat, fname)
    if not match:
        raise ValueError(f"Filename missing parameters: {fname}")

    drive = int(match.group(1))
    tone = int(match.group(2))
    return drive, tone


# ----------------------------------------------------------
# Main evaluation
# ----------------------------------------------------------
def evaluate(weights_path: Path):

    root = load_config()
    distorted_dir = root / "distorted"

    if not distorted_dir.exists():
        raise FileNotFoundError(f"Missing distorted directory: {distorted_dir}")

    # FILTER OUT macOS hidden files and empty files
    wav_files = [
        w for w in distorted_dir.glob("*.wav")
        if not w.name.startswith("._") and w.stat().st_size > 500
    ]

    if not wav_files:
        raise FileNotFoundError(f"No valid WAV files found in {distorted_dir}")

    print(f"Found {len(wav_files)} distorted WAVs.\n")

    # Load model
    model = PedalResNet(output_size=2, use_pretrained=False)
    model.load_weights(weights_path)
    model.eval()

    true_vals = []
    pred_vals = []
    correct_list = []
    predicted_correct_list = []

    for wav in wav_files:
        true_drive, true_tone = parse_filename(wav.name)
        true_vals.append((true_drive, true_tone))

        mel = load_wav_as_mel(wav)

        with torch.no_grad():
            out = model(mel) * 10   # undo training normalization

        pred_drive, pred_tone = out[0].tolist()
        pred_vals.append((pred_drive, pred_tone))

        # "Correct" prediction = within ±5
        is_correct = (
            abs(pred_drive - true_drive) <= 5 and
            abs(pred_tone - true_tone) <= 5
        )
        correct_list.append(1 if is_correct else 0)
        predicted_correct_list.append(1)  # model always gives 1 prediction

        print(f"{wav.name}")
        print(f"  True: drive={true_drive}, tone={true_tone}")
        print(f"  Pred: drive={pred_drive:.1f}, tone={pred_tone:.1f}")
        print(f"  Correct: {is_correct}\n")

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------
    true_drives = np.array([t[0] for t in true_vals])
    pred_drives = np.array([p[0] for p in pred_vals])

    true_tones = np.array([t[1] for t in true_vals])
    pred_tones = np.array([p[1] for p in pred_vals])

    drive_mae = np.mean(np.abs(true_drives - pred_drives))
    tone_mae  = np.mean(np.abs(true_tones - pred_tones))

    drive_rmse = np.sqrt(np.mean((true_drives - pred_drives)**2))
    tone_rmse  = np.sqrt(np.mean((true_tones - pred_tones)**2))

    accuracy = sum(correct_list) / len(correct_list)

    # Confusion matrix (0 = wrong, 1 = correct)
    cm = confusion_matrix(correct_list, correct_list)

    # ------------------------------------------------------
    # Print report
    # ------------------------------------------------------
    print("\n================= RESULTS =================")
    print(f"Samples Evaluated: {len(wav_files)}")
    print(f"Accuracy (±5 tolerance): {accuracy*100:.2f}%")
    print("-------------------------------------------")
    print(f"Drive MAE : {drive_mae:.2f}")
    print(f"Drive RMSE: {drive_rmse:.2f}")
    print(f"Tone  MAE : {tone_mae:.2f}")
    print(f"Tone  RMSE: {tone_rmse:.2f}")
    print("-------------------------------------------")
    print("Confusion Matrix (Correct vs Incorrect):")
    print(cm)
    print("==========================================\n")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    weights = project_root / "weights" / "guitar_model_mel_36300_100.pth"

    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}")

    evaluate(weights)
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
    weights_path = project_root / "weights" / "guitar_model_mel_36300_100.pth"

    evaluate_model(weights_path, distorted_dir)
