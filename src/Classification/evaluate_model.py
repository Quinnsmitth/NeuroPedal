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
from tqdm import tqdm


# Load WAV -> mono -> pad and trim -> mel db
def load_wav_as_mel(path: Path, target_length: int = 160000):
    try:
        with AudioFile(str(path)) as f:
            audio = f.read(f.frames)
            sr = f.samplerate
    except Exception as e:
        raise RuntimeError(f"Could not read WAV: {path} ({e})")

    audio = torch.tensor(audio, dtype=torch.float32)

    # stereo -> mono
    if audio.ndim == 2:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # pad and trim
    if audio.size(1) > target_length:
        audio = audio[:, :target_length]
    else:
        audio = torch.nn.functional.pad(audio, (0, target_length - audio.size(1)))

    mel = mel_spectrogram(audio)
    return mel.unsqueeze(0)


# Parse true parameters from filename
def parse_drive_tone(filename: str):
    parts = filename.split("_")
    drive = int([p for p in parts if p.startswith("drive")][0].replace("drive", ""))
    tone = int([p for p in parts if p.startswith("tone")][0].replace("tone", ""))
    return drive, tone


# Main evaluation function
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
            pred = model(mel)[0] * 10  # undo normalization
            drive_pred, tone_pred = pred.tolist()

            # Round predictions to nearest 10
            drive_pred_rounded = int(round(drive_pred / 10) * 10)
            tone_pred_rounded = int(round(tone_pred / 10) * 10)

        total_drive_error += abs(drive_pred_rounded - gt_drive)
        total_tone_error += abs(tone_pred_rounded - gt_tone)
        count += 1

        # Print rounded predictions
        print(f"{f.name}")
        print(f"  True: drive={gt_drive}, tone={gt_tone}")
        print(f"  Pred (raw)   : drive={drive_pred:.1f}, tone={tone_pred:.1f}")
        print(f"  Pred (rounded): drive={drive_pred_rounded}, tone={tone_pred_rounded}\n")

    mean_drive_error = total_drive_error / count
    mean_tone_error = total_tone_error / count
    mean_total_error = (mean_drive_error + mean_tone_error) / 2

    # Print final results
    print("\n================= RESULTS =================")
    print(f"Samples Evaluated: {len(wav_files)}")
    print("-------------------------------------------")
    print(f"Mean Drive Error: {mean_drive_error:.3f}")
    print(f"Mean Tone Error : {mean_tone_error:.3f}")
    print(f"Average Error   : {mean_total_error:.3f}")
    print("==========================================\n")

    return mean_drive_error, mean_tone_error, mean_total_error


if __name__ == "__main__":
    root = load_config()  # USB root
    distorted_dir = root / "distorted"

    project_root = Path(__file__).resolve().parents[2]
    weights_path = project_root / "weights" / "guitar_model_mel_36300_100.pth"

    evaluate_model(weights_path, distorted_dir)
