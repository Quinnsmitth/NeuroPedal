from select_path import load_config
from pedalboard import load_plugin
from pedalboard.io import AudioFile
from pedal_init import initialize_plugin
import numpy as np
import os

root = load_config()
input_file = root / "clean/clip_011.wav"
output_dir = root / "distorted"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Input file: {input_file}")
print(f"Output directory: {output_dir}")

plugin_path = "/Library/Audio/Plug-Ins/VST3/Audiority/Dr Drive.vst3"
plugin = initialize_plugin(load_plugin(plugin_path))
print(f"Plugin loaded: {plugin.name}")

drive_vals = np.linspace(0, 100, 11)
tone_vals = np.linspace(0, 100, 11)

# Load the input file
with AudioFile(str(input_file)) as file:
    audio = file.read(file.frames)
    sr = file.samplerate

# Process with all drive/tone combinations
for drive in drive_vals:
    for tone in tone_vals:
        plugin.od_drive = drive
        plugin.od_bright = tone
        effected = plugin(audio, sr)

        out = output_dir / f"{input_file.stem}_drive{drive:.0f}_tone{tone:.0f}.wav"
        with AudioFile(str(out), "w", sr, effected.shape[0]) as o:
            o.write(effected)

print(f"{input_file.name} processed ")
