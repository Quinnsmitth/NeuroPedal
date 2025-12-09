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
        audio = torch.nn.functional.pad(audio, (0, target_length - audio.size(1)))

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

    for f in wav_files:
        # Skip hidden or macOS metadata files
        if f.name.startswith(".") or f.name.startswith("._"):
            continue

        try:
            mel = load_wav_as_mel(f)  # (1, 1, n_mels, time)
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

    true_drive = np.array(true_drive, dtype=float)
    true_tone = np.array(true_tone, dtype=float)
    pred_drive = np.array(pred_drive, dtype=float)
    pred_tone = np.array(pred_tone, dtype=float)

    return true_drive, pred_drive, true_tone, pred_tone


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
    out_path = out_dir / f"scatter_true_vs_pred_{name.lower()}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved scatter plot: {out_path}")


def plot_error_histogram(y_true, y_pred, name, out_dir: Path):
    errors = np.abs(y_pred - y_true)
    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=20, alpha=0.8)
    plt.xlabel(f"|Prediction Error| ({name})")
    plt.ylabel("Count")
    plt.title(f"{name} Absolute Error Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f"hist_errors_{name.lower()}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved error histogram: {out_path}")


def plot_metric_table(metrics_dict, out_dir: Path):
    """
    metrics_dict example:
    {
        "Drive": {"MAE": ..., "MSE": ..., "R2": ...},
        "Tone":  {"MAE": ..., "MSE": ..., "R2": ...}
    }
    """
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis("off")

    rows = []
    for key, vals in metrics_dict.items():
        rows.append(
            [
                key,
                f"{vals['MAE']:.3f}",
                f"{vals['MSE']:.3f}",
                f"{vals['R2']:.3f}",
            ]
        )

    table = ax.table(
        cellText=rows,
        colLabels=["Target", "MAE", "MSE", "R²"],
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 2)
    plt.title("Regression Metrics Summary", pad=10)
    plt.tight_layout()

    out_path = out_dir / "metrics_table.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved metrics table: {out_path}")


def plot_training_curves_if_available(out_dir: Path):
    """
    Optional: plot training/validation loss curves.

    Expects a file like:
        project_root / "weights" / "training_history.npz"
    with arrays: 'epochs', 'train_loss', 'val_loss'

    If file is missing, we just print a message and skip.
    """
    history_path = project_root / "weights" / "training_history.npz"
    if not history_path.exists():
        print(f"[INFO] No training history found at {history_path}, "
              f"skipping training curve plot.")
        return

    data = np.load(history_path)
    epochs = data["epochs"]
    train_loss = data["train_loss"]
    val_loss = data["val_loss"]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "training_validation_loss.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved training/validation loss curve: {out_path}")


# ------------------------------
# MAIN
# ------------------------------
def main():
    root = load_config()
    distorted_dir = root / "distorted"

    weights_path = project_root / "weights" / "guitar_model_mel_36300_100.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Where to save plots
    out_dir = project_root / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Run model on dataset & collect predictions
    true_drive, pred_drive, true_tone, pred_tone = run_inference_over_folder(
        weights_path, distorted_dir
    )

    # 2) Compute metrics
    print("\n=== METRICS ===")
    drive_mae, drive_mse, drive_r2 = compute_metrics(
        true_drive, pred_drive, label="Drive"
    )
    tone_mae, tone_mse, tone_r2 = compute_metrics(
        true_tone, pred_tone, label="Tone"
    )

    metrics = {
        "Drive": {"MAE": drive_mae, "MSE": drive_mse, "R2": drive_r2},
        "Tone": {"MAE": tone_mae, "MSE": tone_mse, "R2": tone_r2},
    }

    # 3) Plots

    # Scatter plots
    plot_scatter_true_vs_pred(true_drive, pred_drive, "Drive", out_dir)
    plot_scatter_true_vs_pred(true_tone, pred_tone, "Tone", out_dir)

    # Error histograms
    plot_error_histogram(true_drive, pred_drive, "Drive", out_dir)
    plot_error_histogram(true_tone, pred_tone, "Tone", out_dir)

    # Metrics table figure
    plot_metric_table(metrics, out_dir)

    # Training vs validation loss (optional)
    plot_training_curves_if_available(out_dir)

    print("\nAll plots saved to:", out_dir)


if __name__ == "__main__":
    main()
