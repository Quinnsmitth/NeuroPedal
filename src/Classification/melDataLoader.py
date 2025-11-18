# src/Classification/dataLoader.py
import os
import torch
from torch.utils.data import Dataset
from pedalboard.io import AudioFile
import numpy as np  # (not strictly needed, but kept in case you add augments)
from melSpec import mel_spectrogram


class GuitarPedalDataset(Dataset):
    """
    Dataset for distorted WAV files with filenames encoding drive/tone, e.g.:

        clean_riff_005_drive40_tone50.wav

    Returns:
        mel_db: Tensor (1, 128, time)
        label: Tensor([drive, tone]) as floats
    """

    def __init__(self, data_dir, transform=None, target_length=160000):
        # 4 seconds at ~40kHz
        self.data_dir = data_dir
        self.transform = transform
        self.target_length = target_length

        self.files = [
            f
            for f in os.listdir(data_dir)
            if f.endswith(".wav") and not f.startswith("._")
        ]

    def __len__(self):
        return len(self.files)

    def pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        """Ensure all waveforms have the same length along time dimension."""
        length = waveform.size(1)

        if length > self.target_length:
            waveform = waveform[:, : self.target_length]
        elif length < self.target_length:
            pad_amount = self.target_length - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        return waveform

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)

        try:
            # Use pedalboard for robust reading
            with AudioFile(file_path) as f:
                audio = f.read(f.frames)
                sr = f.samplerate
        except Exception as e:
            print(f"[WARN] Skipping unreadable file: {file_name} ({e})")
            # Dummy fallback (shape roughly matching a typical mel)
            return (
                torch.zeros((1, 128, 313), dtype=torch.float32),
                torch.tensor([0.0, 0.0], dtype=torch.float32),
            )

        # Convert to tensor, mono, fixed length
        audio = torch.tensor(audio, dtype=torch.float32)

        if audio.ndim == 2:  # (channels, N) → mono
            audio = torch.mean(audio, dim=0, keepdim=True)

        audio = self.pad_or_trim(audio)  # (1, target_length)

        # Canonical mel spectrogram pipeline (shared with inference)
        # NOTE: We use the default sample_rate inside mel_spectrogram (41000),
        # matching your previous training code.
        mel_db = mel_spectrogram(audio)

        # Parse drive / tone from filename
        base = os.path.splitext(file_name)[0]
        parts = base.split("_")
        try:
            drive = int(
                [p for p in parts if p.startswith("drive")][0].replace("drive", "")
            )
            tone = int(
                [p for p in parts if p.startswith("tone")][0].replace("tone", "")
            )
        except Exception:
            drive, tone = 0, 0

        label = torch.tensor([drive, tone], dtype=torch.float32)

        return mel_db, label
