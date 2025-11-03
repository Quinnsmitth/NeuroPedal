import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import GuitarPedalDataset
from model import WaveNet  # explicitly import WaveNet
from fileUtil import get_file_name
from fileLoader import getData
def train_model(clean_dir, dist_dir, chunk_size=44100, batch_size=8, epochs=5, lr=1e-3, device=None):
    # Use GPU if available
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset and dataloader
    dataset = GuitarPedalDataset(clean_dir, dist_dir, chunk_size=chunk_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = WaveNet(layers=12, channels=64).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    all_losses = []

    # Training loop
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        for i, (y_clean, y_dist) in enumerate(dataloader):  # only 2 values returned
            y_clean = y_clean.to(device)
            y_dist = y_dist.to(device)

            optimizer.zero_grad()
            output = model(y_clean)
            loss = criterion(output, y_dist)
            loss.backward()
            optimizer.step()

            all_losses.append(loss.item())
            print(f"Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.6f}")
    file_name = get_file_name("weights/", "model", ".pth")
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")


    return model, all_losses

# Example usage:
if __name__ == "__main__":
    clean_dir = getData("clean")
    dist_dir = getData("dist")
    model, losses = train_model(clean_dir, dist_dir)
