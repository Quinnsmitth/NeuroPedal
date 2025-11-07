import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from dataLoader import GuitarPedalDataset
from src.select_path import load_config
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

root = load_config()
dist = root / "distorted"

def train_model(data_dir, num_epochs=50, batch_size=8, lr=1e-4, model_name="resnet18"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = GuitarPedalDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Training on {len(dataset)} samples for {num_epochs} epochs")

    # --- Load pretrained model ---
    if model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Final regression layer → 2 outputs (drive, tone)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # Loss + Optimizer + LR Scheduler
    criterion = nn.SmoothL1Loss()  # <-- Better for noisy regression
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)

            if x.ndim == 3:
                x = x.unsqueeze(1)

            # Normalize spectrogram (per-sample)
            mean = x.mean(dim=[1, 2, 3], keepdim=True)
            std = x.std(dim=[1, 2, 3], keepdim=True) + 1e-6
            x = (x - mean) / std

            # Convert 1-channel spectrogram to 3-channel to use pretrained weights
            x = x.repeat(1, 3, 1, 1)

            # Normalize knob labels from 0–10 → 0–1 range
            y = y / 10.0

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(loader)
        scheduler.step(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] — Avg Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "../../weights/guitar_model_improved.pth")
    print("\nTraining complete — model saved as guitar_model_improved.pth\n")


if __name__ == "__main__":
    train_model(data_dir=dist, model_name="resnet34")  # try resnet18 or resnet34
