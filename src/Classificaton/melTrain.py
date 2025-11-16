import sys
from pathlib import Path

# Add /src directory to Python path BEFORE other imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torchvision import models
import torchaudio
import numpy as np
from melDataLoader import mel_spectrogram  # If your loader has a mel function
from select_path import load_config


def load_model(weights_path):
    """Load a ResNet34 model trained on 1-channel mel spectrograms."""
    model = models.resnet34(weights=None)

    # 1-channel for mel spectrograms
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # 2 output values (Drive, Tone)
    model.fc = nn.Linear(model.fc.in_features, 2)

    # Load checkpoint
    state = torch.load(weights_path, map_location="cpu")

    # Filter out conv1 mismatch if checkpoint used 3 channels
    filtered_state = {}
    for k, v in state.items():
        if k == "conv1.weight" and v.shape[1] != 1:
            print(f"[INFO] Skipping conv1 mismatch: checkpoint conv1 = {v.shape}")
            continue
        filtered_state[k] = v

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval()
    return model


def preprocess_audio(wav_path, target_length=160000, sr=40000):
    """Load WAV → convert to mel spectrogram identical to training."""
    waveform, file_sr = torchaudio.load(str(wav_path))

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to match training
    if file_sr != sr:
        resampler = torchaudio.transforms.Resample(file_sr, sr)
        waveform = resampler(waveform)

    # Pad or trim to target_length (same as dataset)
    if waveform.shape[1] < target_length:
        pad = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :target_length]

    # Convert to mel
    mel = mel_spectrogram(waveform)  # Uses your same loader function

    # ResNet expects (B, C, H, W)
    mel = mel.unsqueeze(0)  # add batch dimension
    return mel


def predict(wav_path, weights_path):
    """Run inference on a single wav file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(weights_path)
    model = model.to(device)

    mel = preprocess_audio(wav_path)  # shape (1, 1, mel_bins, time)
    mel = mel.to(device)

    with torch.no_grad():
        output = model(mel)

    # Output was normalized during training: y / 10.
    output = output * 10.0

    drive = float(output[0, 0].cpu())
    tone = float(output[0, 1].cpu())

    print(f"\nPrediction for {wav_path}:")
    print(f"  Drive: {drive:.2f}")
    print(f"  Tone:  {tone:.2f}")

    return drive, tone


if __name__ == "__main__":
    root = load_config()
    weights_path = "/Users/quinnsmith/Desktop/NeuroPedal-1/weights"

    # Example usage:
    test_wav = root/"train/riff_drive50_tone100.wav"
    predict(test_wav, weights_path)
