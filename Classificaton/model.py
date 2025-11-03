import torch
import torch.nn as nn
import torchvision.models as models

class PedalResNet(nn.Module):
    def __init__(self):
        super(PedalResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)  # Drive, Tone

    def forward(self, x):
        return self.resnet(x)
