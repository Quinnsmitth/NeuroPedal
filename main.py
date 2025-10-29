import torch
from trainer import train_model
import matplotlib.pyplot as plt

clean_dir = "/Users/quinnsmith/Desktop/guitar_data/clean"
dist_dir = "/Users/quinnsmith/Desktop/guitar_data/dist"

def plotLoss(losses):
    plt.figure(figsize=(10,4))
    plt.plot(losses)
    plt.xlabel("Batch")
    plt.ylabel("MSE Loss")
    plt.title("WaveNet Training Loss")
    plt.show()

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, losses = train_model(clean_dir, dist_dir, chunk_size=44100, batch_size=8, epochs=5, device=device)
    return model, losses

model, losses = run_training()
#plotLoss(losses)
