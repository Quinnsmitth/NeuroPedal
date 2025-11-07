import os
import torch
import torchaudio
from torch.utils.data import Dataset

class GuitarPedalDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_length=160000):  # ≈ 4 sec at 40kHz
        self.data_dir = data_dir
        self.transform = transform
        self.target_length = target_length
        self.files = [
            f for f in os.listdir(data_dir)
            if f.endswith(".wav") and not f.startswith("._")
        ]

    def __len__(self):
        return len(self.files)

    def pad_or_trim(self, waveform):
        """Ensure all waveforms have the same length"""
        length = waveform.size(1)
        if length > self.target_length:
            waveform = waveform[:, :self.target_length]
        elif length < self.target_length:
            pad_amount = self.target_length - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        return waveform

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)

        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            print(f" Skipping unreadable file: {file_name} ({e})")
            return torch.zeros((2, self.target_length)), torch.tensor([0, 0], dtype=torch.float32)

        waveform = self.pad_or_trim(waveform)
        if self.transform:
            waveform = self.transform(waveform)

        # Parse drive and tone
        base = os.path.splitext(file_name)[0]
        parts = base.split('_')
        drive_part = [p for p in parts if p.startswith("drive")]
        tone_part = [p for p in parts if p.startswith("tone")]

        try:
            drive = int(drive_part[0].replace("drive", "")) if drive_part else 0
            tone = int(tone_part[0].replace("tone", "")) if tone_part else 0
        except Exception as e:
            print(f" Skipping malformed file: {file_name} ({e})")
            drive, tone = 0, 0

        label = torch.tensor([drive, tone], dtype=torch.float32)
        return waveform, label
