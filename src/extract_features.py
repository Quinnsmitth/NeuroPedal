# src/extract_features.py

import torch
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm          # <-- progress bar
from select_path import load_config

# Load root (USB or other)
root = load_config()

distorted_dir = root / "distorted"
feature_file = root / "features" / "distorted_features.csv"
feature_file.parent.mkdir(parents=True, exist_ok=True)

# Torchaudio MFCC extractor
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=41000,
    n_mfcc=13,
    melkwargs={
        "n_fft": 1024,
        "n_mels": 128,
        "hop_length": 512
    }
)

rows = []
all_files = list(distorted_dir.glob("*.wav"))

print(f"Processing {len(all_files)} WAV files...\n")

# Progress bar
for f in tqdm(all_files, desc="Extracting MFCC features"):
    try:
        # Load WAV
        waveform, sr = torchaudio.load(f)

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != 41000:
            waveform = torchaudio.functional.resample(waveform, sr, 41000)
            sr = 41000

        # Compute MFCCs
        mfcc = mfcc_transform(waveform)    # (1, 13, time)
        mfcc_mean = mfcc.mean(dim=2).squeeze(0).numpy()  # (13,)

        # Parse drive/tone
        parts = f.stem.split("_")
        drive = float([p for p in parts if p.startswith("drive")][0].replace("drive", ""))
        tone  = float([p for p in parts if p.startswith("tone")][0].replace("tone", ""))

        rows.append([*mfcc_mean, drive, tone])

    except Exception as e:
        print(f"[WARN] Skipping file {f}: {e}")

# Save CSV
columns = [f"mfcc_{i}" for i in range(13)] + ["drive", "tone"]
df = pd.DataFrame(rows, columns=columns)
df.to_csv(feature_file, index=False)

print(f"\nFeatures saved to {feature_file}")
