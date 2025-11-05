import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataLoader import GuitarPedalDataset
from model import PedalResNet

def train_model(data_dir, epochs=20, batch_size=8, lr=1e-4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GuitarPedalDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PedalResNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training on {len(dataset)} samples for {epochs} epochs")
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if x.ndim == 3:
                x = x.unsqueeze(1)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "ts9_resnet_weights.pth")
    print("Training complete, model saved as ts9_resnet_weights.pth")

if __name__ == "__main__":
    train_model(data_dir="/Volumes/PortableSSD/data/split")
