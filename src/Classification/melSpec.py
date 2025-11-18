# src/Classification/melSpec.py
import torch
import torchaudio


def mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 41000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
) -> torch.Tensor:
    """
    Convert a waveform tensor to a normalized mel spectrogram.

    Args:
        waveform (Tensor): Audio tensor, shape (1, N)
        sample_rate (int): Sample rate assumed for the waveform
        n_fft (int): FFT window size
        hop_length (int): Hop length
        n_mels (int): Number of mel bins

    Returns:
        Tensor: Normalized mel spectrogram, shape (1, n_mels, time)
    """

    # 1. Mel spectrogram (power)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel = mel_transform(waveform)  # (1, n_mels, time)

    # 2. Convert amplitude to dB
    db_transform = torchaudio.transforms.AmplitudeToDB()
    mel_db = db_transform(mel)

    # 3. Per-spectrogram normalization (mean=0, std=1)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    return mel_db
