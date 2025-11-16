#src/Classification/model.py
import torch
import torch.nn as nn
import torchvision.models as models

class PedalResNet(nn.Module):
    def __init__(self, output_size=2):
        super().__init__()
        self.resnet = models.resnet34(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_size)

    def forward(self, x):
        return self.resnet(x)

    def load_weights(self, path):
        state = torch.load(path, map_location="cpu")
        filtered_state = {}
        for k, v in state.items():
            if k == "conv1.weight" and v.shape != self.resnet.conv1.weight.shape:
                print(f"[INFO] Skipping conv1 mismatch: {v.shape} -> {self.resnet.conv1.weight.shape}")
                continue
            filtered_state[k] = v
        missing, unexpected = self.resnet.load_state_dict(filtered_state, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        self.eval()
