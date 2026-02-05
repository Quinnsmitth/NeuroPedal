# visualize_results.py
"""
Visualize NeuroPedal model performance.

Generates:
  1) Scatter plots: true vs predicted (drive & tone)
  2) Error histograms: |pred - true| for drive & tone
  3) Metrics table figure (MAE, MSE, R²)
  4) Training vs validation loss curves (if history file exists)

Assumptions:
  - You have a trained weights file in weights/
  - Distorted WAV files live in root / "distorted"
  - Filenames encode labels like: ..._drive40_tone50.wav
  - Optional: training history saved as 'training_history.npz'
"""

import sys
from pathlib import Path

import numpy as np
import torch
from pedalboard.io import AudioFile
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from melSpec import mel_spectrogram
from model import PedalResNet

# Make src importable for select_path
project_root = Path(__file__).resolve().parents[2]
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from select_path import load_config  # noqa: E402


# ------------------------------
# AUDIO → MEL HELPERS
# ------------------------------
def load_wav_as_mel(path: Path, target_length: int = 160000):
    """
    Load WAV file, convert to mono, pad/trim to target_length, and
    compute normalized Mel-spectrogram.

    Returns:
        mel tensor of shape (1, 1, n_mels, time)
    """
    try:
        with AudioFile(str(path)) as f:
            audio = f.read(f.frames)
    except Exception as e:
        raise RuntimeError(f"Could not read WAV: {path} ({e})")

    audio = torch.tensor(audio, dtype=torch.float32)

    # Stereo → mono
    if audio.ndim == 2:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Pad or trim
    if audio.size(1) > target_length:
        audio = audio[:, :target_length]
    else:
        audio = torch.nn.functional.pad(
            audio, (0, target_length - audio.size(1))
        )

    mel = mel_spectrogram(audio)  # (1, n_mels, time)

    # Add batch dim: (1, 1, n_mels, time)
    return mel.unsqueeze(0)


def parse_drive_tone(filename_stem: str):
    """
    Extract drive and tone labels from a filename stem like:
        'clean_riff_005_drive40_tone50'
    """
    parts = filename_stem.split("_")
    drive = int([p for p in parts if p.startswith("drive")][0].replace("drive", ""))
    tone = int([p for p in parts if p.startswith("tone")][0].replace("tone", ""))
    return drive, tone


# ------------------------------
# EVALUATION LOOP
# ------------------------------
def run_inference_over_folder(weights_path: Path, distorted_dir: Path):
    """
    Run model over all WAV files in distorted_dir and collect
    true and predicted drive/tone arrays.

    Returns:
        true_drive, pred_drive, true_tone, pred_tone (all np.ndarrays)
    """
    print(f"Loading model from: {weights_path}")

    model = PedalResNet(output_size=2, use_pretrained=False)
    model.load_weights(weights_path)
    model.eval()

    wav_files = sorted(distorted_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {distorted_dir}")

    true_drive, pred_drive = [], []
    true_tone, pred_tone = [], []

    print(f"Evaluating on {len(wav_files)} files...\n")

    for f in tqdm(
        wav_files,
        desc="Running inference",
        unit="file",
        miniters=1,
    ):
        # Skip hidden or macOS metadata files
        if f.name.startswith(".") or f.name.startswith("._"):
            continue

        try:
            mel = load_wav_as_mel(f)
        except RuntimeError as e:
            print(f"Skipping invalid WAV: {f.name} ({e})")
            continue

        gt_drive, gt_tone = parse_drive_tone(f.stem)

        with torch.no_grad():
            out = model(mel)[0] * 10.0  # undo training normalization y/10
            drive_pred, tone_pred = out.tolist()

        true_drive.append(gt_drive)
        true_tone.append(gt_tone)
        pred_drive.append(drive_pred)
        pred_tone.append(tone_pred)

    return (
        np.array(true_drive, dtype=float),
        np.array(pred_drive, dtype=float),
        np.array(true_tone, dtype=float),
        np.array(pred_tone, dtype=float),
    )


# ------------------------------
# METRICS & PLOTS
# ------------------------------
def compute_metrics(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label} MAE: {mae:.3f} | MSE: {mse:.3f} | R²: {r2:.3f}")
    return mae, mse, r2


def plot_scatter_true_vs_pred(y_true, y_pred, name, out_dir: Path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"True vs Predicted {name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"scatter_true_vs_pred_{name.lower()}.png", dpi=200)
    plt.close()


def plot_error_histogram(y_true, y_pred, name, out_dir: Path):
    errors = np.abs(y_pred - y_true)
    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=20, alpha=0.8)
    plt.xlabel(f"|Prediction Error| ({name})")
    plt.ylabel("Count")
    plt.title(f"{name} Absolute Error Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"hist_errors_{name.lower()}.png", dpi=200)
    plt.close()


def plot_metric_table(metrics_dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis("off")

    rows = [
        [k, f"{v['MAE']:.3f}", f"{v['MSE']:.3f}", f"{v['R2']:.3f}"]
        for k, v in metrics_dict.items()
    ]

    ax.table(
        cellText=rows,
        colLabels=["Target", "MAE", "MSE", "R²"],
        loc="center",
        cellLoc="center",
    )
    plt.title("Regression Metrics Summary", pad=10)
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_table.png", dpi=200)
    plt.close()


def plot_training_curves_if_available(out_dir: Path):
    history_path = project_root / "weights" / "training_history.npz"
    if not history_path.exists():
        print("[INFO] No training history found, skipping curve plot.")
        return

    data = np.load(history_path)
    plt.figure(figsize=(6, 4))
    plt.plot(data["epochs"], data["train_loss"], label="Train Loss")
    plt.plot(data["epochs"], data["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_validation_loss.png", dpi=200)
    plt.close()


# ------------------------------
# MAIN
# ------------------------------
def main():
    root = load_config()
    distorted_dir = root / "distorted"

    weights_path = project_root / "weights" / "guitar_model_mel_36300_100.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    out_dir = project_root / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    true_drive, pred_drive, true_tone, pred_tone = run_inference_over_folder(
        weights_path, distorted_dir
    )

    print("\n=== METRICS ===")
    drive_mae, drive_mse, drive_r2 = compute_metrics(true_drive, pred_drive, "Drive")
    tone_mae, tone_mse, tone_r2 = compute_metrics(true_tone, pred_tone, "Tone")

    metrics = {
        "Drive": {"MAE": drive_mae, "MSE": drive_mse, "R2": drive_r2},
        "Tone": {"MAE": tone_mae, "MSE": tone_mse, "R2": tone_r2},
    }

    plot_scatter_true_vs_pred(true_drive, pred_drive, "Drive", out_dir)
    plot_scatter_true_vs_pred(true_tone, pred_tone, "Tone", out_dir)

    plot_error_histogram(true_drive, pred_drive, "Drive", out_dir)
    plot_error_histogram(true_tone, pred_tone, "Tone", out_dir)

    plot_metric_table(metrics, out_dir)
    plot_training_curves_if_available(out_dir)

    print("\nAll plots saved to:", out_dir)


if __name__ == "__main__":
    main()
