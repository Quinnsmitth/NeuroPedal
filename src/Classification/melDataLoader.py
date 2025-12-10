import os
import torch
from torch.utils.data import Dataset
from pedalboard.io import AudioFile
import numpy as np  
from melSpec import mel_spectrogram

# Custom PyTorch Dataset for guitar pedal audio
class JerryGuitarPedalDataset(Dataset):
    """
    Dataset for distorted WAV files whose filenames encode
    pedal settings (drive and tone), for example:

        Jerry_Garcia_riff_005_drive40_tone50.wav

    Each dataset item returns:
        mel_db : Mel spectrogram tensor of shape (1, 128, time)
        label  : Tensor([drive, tone]) as floating-point values
    """

    def __init__(self, data_dir, transform=None, target_length=160000): #3.63 seconds if at 44.1k
        """
        arguments:
            data_dir: Directory containing distorted WAV files
            transform: Optional transform applied to the data
            target_length: Fixed waveform length in samples
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_length = target_length

        # Collect all valid WAV files, ignoring macOS metadata files
        self.files = [
            f
            for f in os.listdir(data_dir)
            if f.endswith(".wav") and not f.startswith("._")
        ]

    def __len__(self):
        """
        Return the number of audio files in the dataset.
        Required by pytorch's Dataset interface.
        """
        return len(self.files)

    def pad_or_trim(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Ensure all audio waveforms have the same length by
        trimming longer files or padding shorter ones with zeros.

        arguments:
            waveform: Tensor of shape (1, N)

        reuturns:
            Tensor of shape (1, target_length)
        """
        length = waveform.size(1)

        if length > self.target_length:
            # Trim waveform if too long
            waveform = waveform[:, : self.target_length]
        elif length < self.target_length:
            # Pad waveform with zeros if too short
            pad_amount = self.target_length - length
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        return waveform

    def __getitem__(self, idx):
        """
        Load one audio file, convert it to a Mel spectrogram,
        and extract the corresponding drive and tone labels.

        arguments:
            idx: Index of the sample

        returns:
            mel_db: Mel spectrogram tensor
            label: Tensor containing [drive, tone]
        """
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)

        try:
            # Load audio file
            with AudioFile(file_path) as f:
                audio = f.read(f.frames)
                sr = f.samplerate
        except Exception as e:
            # Handle unreadable files gracefully
            print(f"Skipping unreadable file: {file_name} ({e})")

            # Return a dummy sample to avoid crashing training
            return (
                torch.zeros((1, 128, 313), dtype=torch.float32),
                torch.tensor([0.0, 0.0], dtype=torch.float32),
            )

        # Convert audio to pytorch tensor
        audio = torch.tensor(audio, dtype=torch.float32)

        # Convert stereo audio to mono by averaging channels
        if audio.ndim == 2:  # (channels, samples)
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Pad or trim waveform to a fixed length
        audio = self.pad_or_trim(audio)  # (1, target_length)

        # Convert waveform to Mel spectrogram
        mel_db = mel_spectrogram(audio)

        # Extract drive and tone values from filename
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
            # Fallback if filename format is unexpected
            drive, tone = 0, 0

        # Create label tensor
        label = torch.tensor([drive, tone], dtype=torch.float32)

        return mel_db, label
