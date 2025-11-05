from pedalboard import load_plugin
from pedalboard.io import AudioFile
import numpy as np
import os
from pathlib import Path
from pedal_init import initialize_plugin

# Define base directory on USB drive
usb_root = Path("E:/guitar_data")

# Define input/output directories
input_dir = usb_root / "clean"
output_dir = usb_root / "distorted"
output_dir.mkdir(parents=True, exist_ok=True)

# Load and initialize the plugin
plugin_path = "C:/Program Files/Common Files/VST3/Audiority/Dr Drive.vst3"
plugin = initialize_plugin(load_plugin(plugin_path))

# Sweep through parameter combinations
drive_vals = np.linspace(0.0, 100.0, 5)
tone_vals = np.linspace(0.0, 100.0, 5)

# Verify clean audio files exist
if not input_dir.exists():
    raise FileNotFoundError(f"Input folder not found: {input_dir}")

# Process each file
for audio_file in input_dir.glob("*.wav"):
    with AudioFile(audio_file) as f:
        audio = f.read(f.frames)
        sr = f.samplerate

    for drive in drive_vals:
        for tone in tone_vals:
            plugin.od_drive = drive
            plugin.od_bright = tone

            effected = plugin(audio, sr)

            out_name = f"{audio_file.stem}_drive{drive:.0f}_tone{tone:.0f}.wav"
            out_path = output_dir / out_name

            with AudioFile(out_path, "w", sr, effected.shape[0]) as o:
                o.write(effected)

    print(f"✅ Processed {audio_file.name} to {output_dir}")
