import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from dataLoader import GuitarPedalDataset
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def train_model(data_dir, num_epochs=20, batch_size=8, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = GuitarPedalDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Training on {len(dataset)} samples for {num_epochs} epochs")

    # --- Load pretrained ResNet18 ---
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    #  Modify first conv layer to take 1-channel input instead of 3
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    #  Modify final fully connected layer for 2 outputs (drive, tone)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)

            #  Ensure correct input shape [B, 1, H, W]
            if x.ndim == 3:
                x = x.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] — Avg Loss: {avg_loss:.6f}")

    print("Training complete.")
    torch.save(model.state_dict(), "guitar_model.pth")
    print("Model saved as guitar_model.pth")


if __name__ == "__main__":
    train_model(data_dir="/Volumes/PortableSSD/guitar_data/distorted")
