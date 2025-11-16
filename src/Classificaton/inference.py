import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
from pedalboard.io import AudioFile
import numpy as np

from melDataLoader import GuitarPedalDataset
from torchvision import models


# Build the exact same model

def load_model(weights_path):
    model = models.resnet34(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

# Preprocess a SINGLE waveform like dataset
def preprocess_single_wav(file_path, target_length=160000):
    with AudioFile(file_path) as f:
        audio = f.read(f.frames)
        sr = f.samplerate

    audio = torch.tensor(audio, dtype=torch.float32)

    if audio.ndim == 2:
        audio = torch.mean(audio, dim=0, keepdim=True)

    length = audio.size(1)
    if length > target_length:
        audio = audio[:, :target_length]
    else:
        audio = torch.nn.functional.pad(audio, (0, target_length - length))

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=41000,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        power=2.0
    )

    db_transform = torchaudio.transforms.AmplitudeToDB()

    mel = mel_transform(audio)
    mel_db = db_transform(mel)

    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    return mel_db.unsqueeze(0)


# Main inference function
def predict(file_path, weights_path):
    model = load_model(weights_path)
    x = preprocess_single_wav(file_path)

    with torch.no_grad():
        output = model(x) * 10.0

    drive, tone = output.squeeze().tolist()

    print("\n========= PREDICTION =========")
    print(f"File: {file_path}")
    print(f"Predicted Drive: {drive:.2f}")
    print(f"Predicted Tone:  {tone:.2f}")
    print("================================\n")


# PATH INPUT SECTION
if __name__ == "__main__":

    wav_path = "/Volumes/PortableSSD/guitar_data/test/riff_drive50_tone100.wav"
    weights_path = "guitar_model_improved.pth"

    # Run prediction
    predict(wav_path, weights_path)
