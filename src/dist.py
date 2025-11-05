from select_path import load_config
from pedalboard import load_plugin
from pedalboard.io import AudioFile
from pedal_init import initialize_plugin
from pathlib import Path
import numpy as np
import os

root = load_config()
input_dir = root / "clean"
output_dir = root / "distorted"
output_dir.mkdir(parents=True, exist_ok=True)

plugin_path = "/Library/Audio/Plug-Ins/VST3/Audiority/Dr Drive.vst3"
plugin = initialize_plugin(load_plugin(plugin_path))

drive_vals = np.linspace(0, 100, 5)
tone_vals = np.linspace(0, 100, 5)

for f in input_dir.glob("*.wav"):
    with AudioFile(f) as file:
        audio = file.read(file.frames)
        sr = file.samplerate
    for drive in drive_vals:
        for tone in tone_vals:
            plugin.od_drive = drive
            plugin.od_bright = tone
            effected = plugin(audio, sr)
            out = output_dir / f"{f.stem}_drive{drive:.0f}_tone{tone:.0f}.wav"
            with AudioFile(out, "w", sr, effected.shape[0]) as o:
                o.write(effected)
    print(f"{f.name} processed")
