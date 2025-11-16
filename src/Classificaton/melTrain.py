# src/Classificaton/melTrain.py

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchaudio
from melSpec import mel_spectrogram
from torchvision import models
from src.select_path import load_config

# Add /src directory to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -----------------------------
# Define PedalResNet
# -----------------------------
class PedalResNet(nn.Module):
    def __init__(self, output_size=2):
        super().__init__()
        self.resnet = models.resnet34(weights=None)
        # Match the training checkpoint conv1 (3x3)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_size)

    def forward(self, x):
        return self.resnet(x)

    def load_weights(self, path):
        state = torch.load(path, map_location="cpu")
        filtered_state = {}
        for k, v in state.items():
            if k == "conv1.weight" and v.shape != self.resnet.conv1.weight.shape:
                print(f"[INFO] Skipping conv1 mismatch: {v.shape} -> {self.resnet.conv1.weight.shape}")
                continue
            filtered_state[k] = v
        missing, unexpected = self.resnet.load_state_dict(filtered_state, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        self.eval()

# -----------------------------
# Audio preprocessing
# -----------------------------
def preprocess_audio(wav_path, target_length=160000, sr=41000):
    """Load WAV -> convert to mel spectrogram identical to training."""
    waveform, file_sr = torchaudio.load(str(wav_path))

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to match training
    if file_sr != sr:
        resampler = torchaudio.transforms.Resample(file_sr, sr)
        waveform = resampler(waveform)

    # Pad or trim to target_length
    if waveform.shape[1] < target_length:
        pad = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :target_length]

    # Convert to mel spectrogram
    mel = mel_spectrogram(waveform)  # your loader function
    mel = mel.unsqueeze(0)  # Add batch dimension
    return mel

# -----------------------------
# Prediction
# -----------------------------
def predict(wav_path, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = PedalResNet()
    model.load_weights(weights_path)
    model.to(device)

    # Preprocess audio
    mel = preprocess_audio(wav_path)
    mel = mel.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(mel)

    # Scale output back to Drive/Tone range
    output = output * 10.0
    drive = float(output[0, 0].cpu())
    tone = float(output[0, 1].cpu())

    print(f"\nPrediction for {wav_path}:")
    print(f"  Drive: {drive:.2f}")
    print(f"  Tone:  {tone:.2f}")

    return drive, tone

# -----------------------------
# Run example
# -----------------------------
if __name__ == "__main__":
    root = load_config()
    weights_path = "/Users/quinnsmith/Desktop/NeuroPedal-1/weights/guitar_model_mel.pth"

    # Example WAV
    test_wav = root / "train/riff_drive50_tone100.wav"
    predict(test_wav, weights_path)
