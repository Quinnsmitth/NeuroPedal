# src/render_to_wav.py
from midi2audio import FluidSynth
from pathlib import Path

sf2 = "data/soundfonts/FluidR3_GM.sf2"
fs = FluidSynth(sf2)

midi_dir = Path("data/midi/clean")
wav_dir = Path("data/wav/clean")
wav_dir.mkdir(parents=True, exist_ok=True)

for midi_file in midi_dir.glob("*.mid"):
    wav_path = wav_dir / (midi_file.stem + ".wav")
    fs.midi_to_audio(str(midi_file), str(wav_path))
