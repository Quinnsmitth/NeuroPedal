import sys
from pathlib import Path
import re
import torch
import numpy as np
from pedalboard.io import AudioFile
from melSpec import mel_spectrogram
from model import PedalResNet
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Set up project paths so local modules can be imported
# Get the root of the project (two levels above this file)
project_root = Path(__file__).resolve().parents[2]

# Add the src directory to Python's search path
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from select_path import load_config

# Audio loading and preprocessing
def load_wav_as_mel(path: Path, target_length: int = 160000):
    """
    Load a WAV file then convert it to mono pad or trim it to a
    fixed length, and compute a Mel spectrogram.

    arguments:
        path: Path to the WAV file
        target_length: Desired number of audio samples

    return:
        Mel spectrogram tensor with a batch dimension
    """
    try:
        # Read audio file using pedalboard
        with AudioFile(str(path)) as f:
            audio = f.read(f.frames)
    except Exception as e:
        # Throw error if the WAV file cannot be read
        raise RuntimeError(f"Could not read WAV: {path} ({e})")

    # Convert audio to pytorch tensor
    audio = torch.tensor(audio, dtype=torch.float32)

    # Convert stereo audio to mono by averaging channels
    if audio.ndim == 2:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Trim audio if longer than target length
    if audio.size(1) > target_length:
        audio = audio[:, :target_length]
    else:
        # Pad audio with zeros if it is too short
        audio = torch.nn.functional.pad(audio, (0, target_length - audio.size(1)))

    # Convert waveform to Mel spectrogram
    mel = mel_spectrogram(audio)

    # Add batch dimension for model input
    return mel.unsqueeze(0)

# Get actual values of Tone and Drive from filename
def parse_drive_tone(filename: str):
    """
    Extract drive and tone values encoded in the filename.

    Example filename format:
        JerryGarcia_drive30_tone70.wav

    reuturns:
        drive (int), tone (int)
    """
    parts = filename.split("_")
    drive = int([p for p in parts if p.startswith("drive")][0].replace("drive", ""))
    tone = int([p for p in parts if p.startswith("tone")][0].replace("tone", ""))
    return drive, tone


# Model evaluation loop
def evaluate_model(weights_path: Path, distorted_dir: Path):
    """
    Load a trained PedalResNet model and evaluate it on
    all valid WAV files in the given directory.

    arguments:
        weights_path: Path to model weights
        distorted_dir: Directory containing test WAV files

    returns:
        Mean drive error, mean tone error, mean total error
    """
    print(f"Loading model: {weights_path}")

    # Initialize model (2 outputs: drive and tone)
    model = PedalResNet(output_size=2, use_pretrained=False)

    # Load trained weights
    model.load_weights(weights_path)

    # Set model to evaluation mode (disable dropout, etc.)
    model.eval()

    # Collect WAV files from the test directory
    wav_files = sorted(distorted_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {distorted_dir}")

    # Track cumulative absolute errors
    total_drive_error = 0
    total_tone_error = 0
    count = 0

    print(f"\nEvaluating {len(wav_files)} files...\n")

    # Process each file with a progress bar
    for f in tqdm(wav_files, desc="Evaluating"):

        # Skip macOS metadata files and hidden files!!!!!
        if f.name.startswith("._") or f.name.startswith("."):
            continue

        try:
            # Convert WAV to Mel spectrogram
            mel = load_wav_as_mel(f)
        except RuntimeError as e:
            # Skip files that fail to load
            print(f"Skipping invalid WAV: {f.name} ({e})")
            continue

        # Extract true drive and tone values from filename
        gt_drive, gt_tone = parse_drive_tone(f.stem)

        # Disable gradient computation during inference
        with torch.no_grad():
            # Model outputs normalized predictions
            pred = model(mel)[0] * 10  # undo normalization
            drive_pred, tone_pred = pred.tolist()

            # Round predictions to nearest multiple of 10
            drive_pred_rounded = int(round(drive_pred / 10) * 10)
            tone_pred_rounded = int(round(tone_pred / 10) * 10)

        # Accumulate absolute errors
        total_drive_error += abs(drive_pred_rounded - gt_drive)
        total_tone_error += abs(tone_pred_rounded - gt_tone)
        count += 1

        # Print per-file prediction results
        print(f"{f.name}")
        print(f"  True: drive={gt_drive}, tone={gt_tone}")
        print(f"  Pred (raw)   : drive={drive_pred:.1f}, tone={tone_pred:.1f}")
        print(f"  Pred (rounded): drive={drive_pred_rounded}, tone={tone_pred_rounded}\n")

    # Handle case where no valid files were processed
    if count == 0:
        print("Sorry... valid WAV files were processed./n Jerry Garcia is sleeping.")
        return None, None, None

    # Compute mean errors
    mean_drive_error = total_drive_error / count
    mean_tone_error = total_tone_error / count
    mean_total_error = (mean_drive_error + mean_tone_error) / 2

    # Print summarized evaluation results Jerry (Garcia) Cleanly
    print("\n================= RESULTS =================")
    print(f"Samples Evaluated: {count}")
    print("-------------------------------------------")
    print(f"Mean Drive Error: {mean_drive_error:.3f}")
    print(f"Mean Tone Error : {mean_tone_error:.3f}")
    print(f"Average Error   : {mean_total_error:.3f}")
    print("==========================================\n")

    return mean_drive_error, mean_tone_error, mean_total_error

# Script entry point
if __name__ == "__main__":
    # Load project root directory from configuration
    root = load_config()

    # Directory containing distorted test audio
    distorted_dir = root / "distorted"

    # Path to trained model weights
    weights_path = project_root / "weights" / "guitar_model_mel_36300_100.pth"

    # Run evaluation
    evaluate_model(weights_path, distorted_dir)
