import os
import torch
from torch.utils.data import Dataset
import torchaudio

class GuitarPedalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)

        waveform, sr = torchaudio.load(file_path)
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=128)(waveform)
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        # Parse labels (supports ts9_driveX_toneY[_ZZZ].wav)
        parts = file_name.split('_')
        # extracts drive and tone from filename
        drive = int(parts[1].replace('drive', ''))
        tone_part = parts[2].replace('tone', '')
        #checks if there is an additional part after tone if so it removes it
        if '.' in tone_part:
            tone = int(tone_part.split('.')[0])
        else:
            tone = int(tone_part.split('_')[0])
        drive /= 10.0
        tone /= 10.0
        # Apply any additional transforms to the spectrogram that have been done to the dataset
        if self.transform:
            mel_spec = self.transform(mel_spec)

        return mel_spec, torch.tensor([drive, tone], dtype=torch.float32)
