# gui_pyqt_infer.py
import sys
from pathlib import Path
import traceback

# Ensure project root (src/) is importable
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# PyQt5 GUI
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QTextEdit, QHBoxLayout, QMessageBox
)

# ML libs
import torch
import torch.nn as nn
from torchvision import models
import torchaudio
import numpy as np

# Your project's functions
from melDataLoader import mel_spectrogram


# ---------------------------
# Helper: load model function
# ---------------------------
def load_model(weights_path: Path, num_outputs: int):
    """Load ResNet34 adapted for 1-channel mel input and load weights."""
    model = models.resnet34(weights=None)  # <-- match training architecture

    # Adapt first conv layer for 1-channel input
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    # Adapt final fully connected layer for 2 outputs
    model.fc = nn.Linear(model.fc.in_features, num_outputs)

    # Load weights
    state = torch.load(str(weights_path), map_location="cpu")
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state_dict = state["model"]
    else:
        state_dict = state

    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------
# Helper: preprocess audio
# ---------------------------
def preprocess_audio(path: Path, cfg: dict):
    """Load WAV and return mel tensor shaped (1, 1, mel_bins, time)."""
    waveform, sr = torchaudio.load(str(path))  # waveform: (channels, samples)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # stereo → mono

    # Resample if needed
    target_sr = cfg.get("sample_rate", None)
    if target_sr and sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
        sr = target_sr

    # Convert to mel spectrogram
    try:
        mel = mel_spectrogram(
            waveform,
            sample_rate=sr,
            n_fft=cfg.get("n_fft", 2048),
            hop_length=cfg.get("hop_length", 512),
            n_mels=cfg.get("n_mels", 128)
        )
    except TypeError:
        mel = mel_spectrogram(waveform)

    # Ensure shape is (1, 1, mel_bins, time)
    if mel.ndim == 3:
        mel = mel.unsqueeze(0)
    elif mel.ndim == 2:
        mel = mel.unsqueeze(0).unsqueeze(0)
    if mel.shape[1] != 1:
        if mel.ndim == 3:
            mel = mel.unsqueeze(1)
    return mel.float()


# ---------------------------
# Helper: prediction logic
# ---------------------------
def run_prediction(model: torch.nn.Module, mel_tensor: torch.Tensor):
    with torch.no_grad():
        out = model(mel_tensor)

    out_np = out.cpu().numpy()
    outputs = out_np[0]

    result = {}
    if outputs.size == 2:
        # Regression: drive & tone
        drive, tone = float(outputs[0]) * 10.0, float(outputs[1]) * 10.0
        result["type"] = "regression"
        result["drive"] = int(round(drive))
        result["tone"] = int(round(tone))
        result["raw"] = [drive, tone]
    else:
        # Classification (fallback)
        probs = torch.softmax(torch.from_numpy(outputs).unsqueeze(0), dim=1).numpy()[0]
        cls = int(np.argmax(probs))
        result["type"] = "classification"
        result["class"] = cls
        result["probs"] = probs.tolist()
    return result


# ---------------------------
# PyQt5 GUI
# ---------------------------
class InferenceWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guitar Pedal Predictor")
        self.setMinimumSize(560, 360)
        self.setAcceptDrops(True)

        # Dark theme styling
        self.setStyleSheet("""
            QWidget { background-color: #0f1115; color: #e6e6e6; font-family: Arial; }
            QLabel#title { font-size: 18px; font-weight: 600; color: #ffffff; }
            QPushButton { background-color: #1f6feb; color: white; border-radius: 6px; padding: 8px 12px; }
            QTextEdit { background-color: #0b0c0f; color: #e6e6e6; border: 1px solid #222; }
            QFrame#drop { border: 2px dashed #333; background-color: #0b0c0f; }
        """)

        layout = QVBoxLayout()
        self.setLayout(layout)

        title = QLabel("Guitar Pedal Predictor — Drop a WAV file or click Choose File")
        title.setObjectName("title")
        layout.addWidget(title)

        # Drop area
        self.drop_frame = QtWidgets.QFrame()
        self.drop_frame.setObjectName("drop")
        self.drop_frame.setFixedHeight(160)
        drop_layout = QHBoxLayout(self.drop_frame)
        self.drop_label = QLabel("Drop WAV file here")
        self.drop_label.setAlignment(QtCore.Qt.AlignCenter)
        drop_layout.addWidget(self.drop_label)
        layout.addWidget(self.drop_frame)

        # Buttons
        btn_layout = QHBoxLayout()
        self.choose_btn = QPushButton("Choose WAV File")
        self.choose_btn.clicked.connect(self.choose_file)
        btn_layout.addWidget(self.choose_btn)

        self.clear_btn = QPushButton("Clear Output")
        self.clear_btn.clicked.connect(self.clear_output)
        btn_layout.addWidget(self.clear_btn)
        layout.addLayout(btn_layout)

        # Output box
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output, stretch=1)

        # ----------------------------
        # Hardcoded config & model
        # ----------------------------
        self.append_output("Loading model (hardcoded ResNet34)...")
        try:
            self.cfg = {
                "sample_rate": 44100,
                "n_fft": 2048,
                "hop_length": 512,
                "n_mels": 128
            }

            self.weights_path = Path("/Users/quinnsmith/Desktop/NeuroPedal-1/weights/guitar_model_mel_36300_100.pth")
            if not self.weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {self.weights_path}")

            num_outputs = 2
            self.model = load_model(self.weights_path, num_outputs)
            self.append_output(f"Model loaded: {self.weights_path}  (outputs: {num_outputs})\n")
        except Exception as e:
            self.append_output("ERROR loading model:\n" + "".join(traceback.format_exception_only(type(e), e)))
            self.model = None

    def append_output(self, text: str):
        self.output.append(text)
        self.output.verticalScrollBar().setValue(self.output.verticalScrollBar().maximum())

    def clear_output(self):
        self.output.clear()

    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open WAV file", str(project_root), "WAV files (*.wav)")
        if file_path:
            self.handle_file(Path(file_path))

    # Drag & drop events
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        event.accept()

    def dropEvent(self, event: QtGui.QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            self.handle_file(Path(path))

    # Handle file & inference
    def handle_file(self, path: Path):
        try:
            if not path.exists():
                raise FileNotFoundError(str(path))
            self.append_output(f"\nLoaded file: {path}")
            if not self.model:
                raise RuntimeError("Model not loaded; can't run inference.")
            self.append_output("Preprocessing audio...")
            mel = preprocess_audio(path, self.cfg)
            self.append_output(f"Mel shape: {tuple(mel.shape)}")
            self.append_output("Running model...")
            result = run_prediction(self.model, mel)
            self.display_result(result)
        except Exception as e:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "Error", str(e))
            self.append_output("ERROR:\n" + tb)

    def display_result(self, result: dict):
        self.append_output("\n=== Prediction ===")
        if result["type"] == "regression":
            self.append_output(f"Drive: {result['drive']}   Tone: {result['tone']}")
            self.append_output(f"(raw values: {result['raw']})")
        else:
            self.append_output(f"Class: {result['class']}")
            self.append_output(f"Probabilities: {result['probs']}")
        self.append_output("==================\n")


# ---------------------------
# Run the app
# ---------------------------
def main():
    app = QApplication(sys.argv)
    window = InferenceWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
