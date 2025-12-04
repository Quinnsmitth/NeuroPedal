# src/Classification/melTrain_optimized.py
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------
#  CUDA & PERFORMANCE SETTINGS
# ------------------------------------------------------------
import torch
torch.backends.cudnn.benchmark = True        # Auto-tune conv kernels
torch.backends.cudnn.enabled = True          # Full cuDNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
#  Fix Python path
# ------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from melDataLoader import GuitarPedalDataset
from select_path import load_config


# ============================================================
#  TRAINING FUNCTION
# ============================================================
def train_model(
    data_dir,
    num_epochs: int = 100,
    batch_size: int = 32,   # Optimized for RTX 3050
    lr: float = 1e-4,
    model_name: str = "resnet34",
):

    print("\n====================")
    print(" CUDA INFORMATION")
    print("====================")
    print("Torch CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("====================\n")

    # --------------------------------------------------------
    #  Dataset
    # --------------------------------------------------------
    dataset = GuitarPedalDataset(data_dir)
    total_size = len(dataset)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"Dataset split: {train_size} train | {val_size} val | {test_size} test\n")

    # --------------------------------------------------------
    #  DATALOADERS (OPTIMIZED)
    # --------------------------------------------------------
    # Use 4 workers because your Ryzen 3100 has 4 cores
    num_workers = 4

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --------------------------------------------------------
    #  MODEL
    # --------------------------------------------------------
    if model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Replace conv1 to accept 1-channel input
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    # Output = [drive, tone]
    model.fc = nn.Linear(model.fc.in_features, 2)

    model = model.to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=4
    )

    # Mixed Precision scaler
    scaler = torch.cuda.amp.GradScaler()

    print(f"Training on {len(train_set)} samples for {num_epochs} epochs.\n")

    # ============================================================
    #  TRAINING LOOP
    # ============================================================
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if x.ndim == 3:
                x = x.unsqueeze(1)

            optimizer.zero_grad()

            # ----- MIXED PRECISION -----
            with torch.cuda.amp.autocast():
                outputs = model(x)
                loss = criterion(outputs, y / 10.0)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --------------------------------------------------------
        #  VALIDATION
        # --------------------------------------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                if x.ndim == 3:
                    x = x.unsqueeze(1)

                with torch.cuda.amp.autocast():
                    outputs = model(x)
                    loss = criterion(outputs, y / 10.0)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] — "
            f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}"
        )

    # ============================================================
    #  TEST LOOP
    # ============================================================
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if x.ndim == 3:
                x = x.unsqueeze(1)

            with torch.cuda.amp.autocast():
                outputs = model(x)
                loss = criterion(outputs, y / 10.0)

            test_loss += loss.item()

    print(f"\nFinal Test Loss: {test_loss / len(test_loader):.6f}")

    # --------------------------------------------------------
    #  SAVE MODEL
    # --------------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]
    weights_dir = project_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    save_path = weights_dir / "guitar_model_mel_optimized.pth"
    torch.save(model.state_dict(), save_path)

    print(f"\nTraining complete — model saved at {save_path}\n")


if __name__ == "__main__":
    root = load_config()
    dist = root / "distorted"
    train_model(data_dir=dist, model_name="resnet34")
