import os
import torch
import torchaudio
from torch.utils.data import Dataset
from pedalboard.io import AudioFile
import numpy as np

class GuitarPedalDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_length=160000):  # 4 sec at 40kHz
        self.data_dir = data_dir
        self.transform = transform
        self.target_length = target_length
        self.files = [
            f for f in os.listdir(data_dir)
            if f.endswith(".wav") and not f.startswith("._")
        ]

        # Define a mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=41000, 
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            power=2.0
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.files)

    def pad_or_trim(self, waveform):
        #Ensure all waveforms have the same length
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
            with AudioFile(file_path) as f:
                audio = f.read(f.frames)
                sr = f.samplerate
        except Exception as e:
            print(f" Skipping unreadable file: {file_name} ({e})")
            return torch.zeros((1, 128, 313)), torch.tensor([0, 0], dtype=torch.float32)

        # Convert to tensor and mono
        audio = torch.tensor(audio, dtype=torch.float32)
        if audio.ndim == 2:
            audio = torch.mean(audio, dim=0, keepdim=True)
        audio = self.pad_or_trim(audio)

        #  Convert waveform  mel spectrogram 
        mel = self.mel_transform(audio)
        mel_db = self.db_transform(mel)

        # Normalize spectrogram
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        #  Parse Drive / Tone from filename 
        base = os.path.splitext(file_name)[0]
        parts = base.split('_')
        try:
            drive = int([p for p in parts if p.startswith("drive")][0].replace("drive", ""))
            tone = int([p for p in parts if p.startswith("tone")][0].replace("tone", ""))
        except:
            drive, tone = 0, 0

        label = torch.tensor([drive, tone], dtype=torch.float32)

        return mel_db, label
