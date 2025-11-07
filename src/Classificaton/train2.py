import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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

    # --- Load full dataset ---
    dataset = GuitarPedalDataset(data_dir)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    # --- Split into train/val/test ---
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    print(f"Dataset split: {train_size} train | {val_size} val | {test_size} test")

    # --- Create data loaders ---
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # --- Model setup ---
    print(f"Training on {len(train_set)} samples for {num_epochs} epochs")

    if model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    model.fc = nn.Linear(model.fc.in_features, 2)  # Drive, Tone regression
    model = model.to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)

    # --- Training loop ---
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            if x.ndim == 3:
                x = x.unsqueeze(1)

            # Normalize input and labels
            mean = x.mean(dim=[1, 2, 3], keepdim=True)
            std = x.std(dim=[1, 2, 3], keepdim=True) + 1e-6
            x = (x - mean) / std
            x = x.repeat(1, 3, 1, 1)
            y = y / 10.0

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if x.ndim == 3:
                    x = x.unsqueeze(1)

                mean = x.mean(dim=[1, 2, 3], keepdim=True)
                std = x.std(dim=[1, 2, 3], keepdim=True) + 1e-6
                x = (x - mean) / std
                x = x.repeat(1, 3, 1, 1)
                y = y / 10.0

                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] — Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # --- Final test evaluation ---
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            if x.ndim == 3:
                x = x.unsqueeze(1)
            mean = x.mean(dim=[1, 2, 3], keepdim=True)
            std = x.std(dim=[1, 2, 3], keepdim=True) + 1e-6
            x = (x - mean) / std
            x = x.repeat(1, 3, 1, 1)
            y = y / 10.0

            outputs = model(x)
            loss = criterion(outputs, y)
            test_loss += loss.item()

    print(f"\nFinal Test Loss: {test_loss / len(test_loader):.6f}")

    torch.save(model.state_dict(), "guitar_model_improved.pth")
    print("\nTraining complete — model saved as guitar_model_improved.pth\n")


if __name__ == "__main__":
    train_model(data_dir=dist, model_name="resnet34")
