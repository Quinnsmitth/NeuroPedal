# src/Classification/model.py
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Union
from pathlib import Path

# ResNet-based model for pedal parameter regression
class PedalResNet(nn.Module):
    """
    ResNet34 adapted for single-channel Mel spectrogram input
    and two regression outputs: [drive, tone].

    NOTE:
    This architecture exactly matches the one used during
    training in melTrain.py, ensuring compatibility when
    loading saved weights.
    """

    def __init__(self, output_size=2, use_pretrained=False):
        """
        arguments:
            output_size: Number of regression outputs
                         (default = 2 for drive and tone)
            use_pretrained: Whether to load ImageNet-pretrained
                            ResNet34 weights
        """
        super().__init__()

        # Initialize base ResNet model

        if use_pretrained:
            # Load ResNet34 with pretrained ImageNet weights
            base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        else:
            # Initialize ResNet34 with random weights
            base = models.resnet34(weights=None)

        # Modify input layer for Mel spectrograms

        # Replace first convolution to accept 1-channel input
        # instead of the standard 3-channel RGB image input
        base.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Modify output layer for regression

        # Replace the classification head with a regression head
        # that outputs [drive, tone]
        base.fc = nn.Linear(base.fc.in_features, output_size)

        self.resnet = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        arguments:
            x: Input tensor of shape (B, 1, n_mels, time)

        Returns:
            Tensor of shape (B, 2) containing normalized
            predictions [drive, tone].

        NOTE:
        If training normalized targets using y / 10,
        predictions should be multiplied by 10 at inference.
        """
        return self.resnet(x)

    def load_weights(self, path: Union[str, Path], map_location: str = "cpu"):
        """
        Load trained weights from melTrain.py.

        arguments:
            path: Path to the saved model weights
            map_location: Device to map tensors to (CPU by default)

        This function supports weights saved as either:
        - a raw state_dict
        - a dictionary containing a 'state_dict' key
        """

        # Load saved weights from Jerry-disk
        state = torch.load(path, map_location=map_location)

        # Handle checkpoints that wrap the state_dict
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        filtered_state = {}

        # Filter state_dict for compatibility

        for k, v in state.items():
            # Guard against mismatched conv1 layer shapes
            # (should not occur if training and inference match)
            if k == "conv1.weight" and v.shape != self.resnet.conv1.weight.shape:
                print(
                    f"[INFO] Jerry Skipping conv1 mismatch: "
                    f"{v.shape} -> {self.resnet.conv1.weight.shape}"
                )
                continue
            filtered_state[k] = v

        # Load weights with relaxed strictness
        missing, unexpected = self.resnet.load_state_dict(
            filtered_state, strict=False
        )

        print("[PedalResNet] Loaded weights from:", path)
        print("  Missing keys:", missing)
        print("  Unexpected keys:", unexpected)

        # Set model to evaluation mode after loading weights
        self.eval()
        return self
