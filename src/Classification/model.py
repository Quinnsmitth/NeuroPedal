# src/Classification/model.py
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Union
from pathlib import Path


class PedalResNet(nn.Module):
    """
    ResNet34 adapted for 1-channel Mel spectrogram input and 2-value regression output:
    [drive, tone].

    NOTE: This architecture is matched to the one used in melTrain.py.
    """

    def __init__(self, output_size=2, use_pretrained=False):
        super().__init__()

        if use_pretrained:
            base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        else:
            base = models.resnet34(weights=None)

        # Match training conv1: 1-channel, 7x7, stride 2, padding 3
        base.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Match training head: 2 outputs (drive, tone)
        base.fc = nn.Linear(base.fc.in_features, output_size)

        self.resnet = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, 1, n_mels, time)
        Returns: (B, 2) > [drive_norm, tone_norm] in the same scale as training.
        If you trained with y/10, multiply predictions by 10 at inference.
        """
        return self.resnet(x)

    def load_weights(self, path: Union[str, Path], map_location: str = "cpu"):
        """
        Load weights saved from melTrain.py.

        Expects a state_dict from a plain ResNet34 with the same conv1/fc
        modifications ( keys like 'conv1.weight', 'layer1.0.conv1.weight', etc.).
        """
        state = torch.load(path, map_location=map_location)

        # Some tooling saves as {"state_dict": {...}}
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        filtered_state = {}

        for k, v in state.items():
            # conv1 mismatch guard (should not trigger now that arch is aligned)
            if k == "conv1.weight" and v.shape != self.resnet.conv1.weight.shape:
                print(
                    f"[INFO] Skipping conv1 mismatch: "
                    f"{v.shape} -> {self.resnet.conv1.weight.shape}"
                )
                continue
            filtered_state[k] = v

        missing, unexpected = self.resnet.load_state_dict(filtered_state, strict=False)
        print("[PedalResNet] Loaded weights from:", path)
        print("  Missing keys:", missing)
        print("  Unexpected keys:", unexpected)

        self.eval()
        return self
