import librosa, pandas as pd, numpy as np
from pathlib import Path

usb_root = Path("E:/guitar_data")
distorted_dir = usb_root / "distorted"
feature_file = usb_root / "features/distorted_features.csv"
feature_file.parent.mkdir(parents=True, exist_ok=True)

rows = []
for f in distorted_dir.glob("*.wav"):
    y, sr = librosa.load(f, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mean_feats = np.mean(mfcc, axis=1)
    parts = f.stem.split("_")
    drive = float(parts[-2].replace("drive", ""))
    tone = float(parts[-1].replace("tone", ""))
    rows.append([*mean_feats, drive, tone])

df = pd.DataFrame(rows, columns=[f"mfcc_{i}" for i in range(13)] + ["drive", "tone"])
df.to_csv(feature_file, index=False)

print(f"✅ Features saved to {feature_file}")
