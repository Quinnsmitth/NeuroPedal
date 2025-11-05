from pedalboard import load_plugin
from pedalboard.io import AudioFile
import numpy as np
import os
from pedal_init import initialize_plugin
# Load the plugin (must be a VST3, not VST2 / VST)
plugin_path = "/Library/Audio/Plug-Ins/VST3/Audiority/Dr Drive.vst3"

plugin = initialize_plugin(load_plugin(plugin_path))

print(plugin.parameters)

#Input and output paths
input_file = '/Users/quinnsmith/Desktop/guitar_data/clean/clip_011.wav' # We'll figure it out
output_dir = '/Users/quinnsmith/Desktop/guitar_data/tstfile' # We'll figure it out 

# Create the output directory if it doesn’t exist
os.makedirs(output_dir, exist_ok=True)

# Load the input audio file
with AudioFile(input_file) as f: # Opens audio file located at input_file, AudioFile is a context manager, so it will automatically close the file when done and f represents the opened audio file object
    audio = f.read(f.frames)
    samplerate = f.samplerate

# Loop through parameter combinations
# Loop through parameter combinations
for drive in np.linspace(0.0, 100.0, 10):
    for tone in np.linspace(0.0, 100.0, 10):
        plugin.od_drive = drive
        plugin.od_bright = tone


        # Process the audio
        effected = plugin(audio, samplerate)

        # Save to a new file
        name = f"drive{drive:.1f}_tone{tone:.1f}.wav"
        path = os.path.join(output_dir, name)
        with AudioFile(path, "w", samplerate, effected.shape[0]) as o:
            o.write(effected)
