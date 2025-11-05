# src/extract_features.py

import librosa
import pandas as pd
import numpy as np
from pathlib import Path
from select_path import load_config   # <-- NEW

# Load the correct base folder (USB / SSD / Desktop / etc.)
root = load_config()

distorted_dir = root / "distorted"
feature_file = root / "features" / "distorted_features.csv"
feature_file.parent.mkdir(parents=True, exist_ok=True)

rows = []

for f in distorted_dir.glob("*.wav"):
    # Load audio
    y, sr = librosa.load(f, sr=None)

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mean_feats = np.mean(mfcc, axis=1)  # shape: (13,)

    # --- Robust filename parsing ---
    # Example filename: "clip_011_drive25_tone60.wav"
    parts = f.stem.split("_")
    drive_val = next(p for p in parts if p.startswith("drive"))
    tone_val = next(p for p in parts if p.startswith("tone"))

    drive = float(drive_val.replace("drive", ""))
    tone = float(tone_val.replace("tone", ""))

    rows.append([*mean_feats, drive, tone])

# Convert to DataFrame
columns = [f"mfcc_{i}" for i in range(13)] + ["drive", "tone"]
df = pd.DataFrame(rows, columns=columns)

# Save CSV
df.to_csv(feature_file, index=False)

print(f"✅ Features saved to {feature_file}")
