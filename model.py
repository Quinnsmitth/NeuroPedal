import torch.nn as nn

class WaveNet(nn.Module):
    def __init__(self, layers=12, channels=64):
        super(WaveNet, self).__init__()
        self.conv1 = nn.Conv1d(1, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x
