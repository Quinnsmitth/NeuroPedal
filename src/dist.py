from select_path import load_config
from pedalboard import load_plugin
from pedalboard.io import AudioFile
from pedal_init import initialize_plugin 
import numpy as np
import os
from tqdm import tqdm  # progress bar

# Configuration
root = load_config()

input_dir = root / "clean"         # Folder containing your input .wav files
output_dir = root / "distorted"    # Folder to save processed files
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")

# Load Plugin
plugin_path = root / "pedal/Dr Drive.vst3"
plugin = load_plugin(str(plugin_path))
#print(f"Plugin loaded: {plugin.name}")

# Parameter Ranges
drive_vals = np.linspace(0, 100, 11)  # 0%, 10%, ..., 100%
tone_vals = np.linspace(0, 100, 11)

# Get valid .wav files 
wav_files = [f for f in input_dir.glob("*.wav") if not f.name.startswith("._")]

# Process Each File
for input_file in wav_files:
    print(f"\n Processing: {input_file.name}")

    # Load the input audio file
    with AudioFile(str(input_file)) as file:
        audio = file.read(file.frames)
        sr = file.samplerate

    # Total combinations
    total_combos = len(drive_vals) * len(tone_vals)

    # Apply plugin across all drive/tone combinations with progress bar
    with tqdm(total=total_combos, desc=f"{input_file.stem}", ncols=80) as pbar:
        for drive in drive_vals:
            for tone in tone_vals:
                plugin.od_drive = drive
                plugin.od_bright = tone
                effected = plugin(audio, sr)

                # Create output filename
                out_file = output_dir / f"{input_file.stem}_drive{drive:.0f}_tone{tone:.0f}.wav"

                # Write processed file
                with AudioFile(str(out_file), "w", sr, effected.shape[0]) as o:
                    o.write(effected)

                pbar.update(1)  # update progress bar

    print(f" Finished: {input_file.name}")

print("\nAll files processed successfully!")
