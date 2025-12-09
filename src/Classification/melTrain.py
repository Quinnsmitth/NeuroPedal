import sys
from pathlib import Path
import warnings

# Allow imports from the project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models 
from tqdm import tqdm

from melDataLoader import GuitarPedalDataset
from select_path import load_config

# Suppress non-critical user warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load dataset path from configuration

root = load_config()
dist = root / "distorted"

# Model training function

def train_model(
    data_dir,
    num_epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    model_name: str = "resnet34",
):
    """
    Train a ResNet-based model to predict Jerry Garcia guitar pedals
    drive and tone values from Mel spectrogram inputs.

    arguments:
        data_dir: Directory containing training WAV files
        num_epochs: Number of training epochs
        batch_size: Number of samples per batch
        lr: Learning rate for the optimizer
        model_name: Which ResNet architecture to use
    """

    # Print CUDA / GPU information for debugging

    print(torch.version.cuda)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.backends.cudnn.version:", torch.backends.cudnn.version())

    # Select GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(0))

    print(f"Using device: {device}\n")

    # Load full dataset

    dataset = GuitarPedalDataset(data_dir)
    total_size = len(dataset)

    # Split dataset into training, validation, and test sets
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    # Use a fixed random seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"Dataset split: {train_size} train | {val_size} val | {test_size} test")

    # Create DataLoaders for batching and shuffling
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Training on {len(train_set)} samples for {num_epochs} epochs")

    # Initialize pretrained ResNet model

    if model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Replace first convolution to accept single-channel input
    model.conv1 = nn.Conv2d(
        1,              # 1 input channel for Mel spectrograms
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )

    # Replace final fully connected layer to output drive and tone
    model.fc = nn.Linear(model.fc.in_features, 2)  # [drive, tone]

    model = model.to(device)

    # Loss function, optimizer, and learning rate scheduler

    # Smooth L1 loss is robust to outliers
    criterion = nn.SmoothL1Loss()

    # Adam optimizer for adaptive learning rates
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Reduce learning rate if validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=4
    )

    # Training loop

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Iterate over training batches
        for x, y in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        ):
            x, y = x.to(device), y.to(device)

            # Ensure input has channel dimension
            # (B, n_mels, time) -> (B, 1, n_mels, time)
            if x.ndim == 3:
                x = x.unsqueeze(1)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(x)

            # Normalize targets to match network output scale
            loss = criterion(outputs, y / 10.0)

            # Backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

    
        # Validation loop
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                if x.ndim == 3:
                    x = x.unsqueeze(1)

                outputs = model(x)
                loss = criterion(outputs, y / 10.0)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] — "
            f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}"
        )

    # Final evaluation on test set

    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            if x.ndim == 3:
                x = x.unsqueeze(1)

            outputs = model(x)
            loss = criterion(outputs, y / 10.0)
            test_loss += loss.item()

    print(f"\nFinal Test Loss: {test_loss / len(test_loader):.6f}")

    # Save trained model weights
    project_root = Path(__file__).resolve().parents[2]
    weights_dir = project_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    save_path = weights_dir / "guitar_model_mel_36300_100.pth"
    torch.save(model.state_dict(), save_path)

    print(f"\nTraining complete — model saved as {save_path}\n")


# Script entry point

if __name__ == "__main__":
    train_model(data_dir=dist, model_name="resnet34")
