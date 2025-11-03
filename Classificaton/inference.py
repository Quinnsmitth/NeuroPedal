import torch
import torchaudio
from model import PedalResNet

model = PedalResNet()
model.load_state_dict(torch.load("ts9_resnet_weights.pth", map_location="cpu"))
model.eval()

def predict_settings(file_path):
    waveform, sr = torchaudio.load(file_path)
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=128)(waveform)
    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec).unsqueeze(0)
    if mel_spec.ndim == 3:
        mel_spec = mel_spec.unsqueeze(1)

    with torch.no_grad():
        pred = model(mel_spec)
    drive = pred[0,0].item() * 10
    tone = pred[0,1].item() * 10
    print(f"Predicted Drive: {drive:.1f}/10, Tone: {tone:.1f}/10")

# Example usage
if __name__ == "__main__":
    predict_settings("data/split/ts9_drive5_tone3_000.wav")
