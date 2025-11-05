# src/generate_midi.py

from mido import MidiFile, MidiTrack, Message
from pathlib import Path
import random
from select_path import load_config

# Load correct dataset root
root = load_config()

midi_dir = root / "midi"
midi_dir.mkdir(parents=True, exist_ok=True)

print(f"Writing MIDI files to: {midi_dir}")

def create_midi(filepath, program=24, length=16):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Set guitar instrument (General MIDI Patch 24 = Nylon Guitar)
    track.append(Message('control_change', control=0, value=0, time=0))  # CC0 Bank MSB = 0
    track.append(Message('control_change', control=32, value=0, time=0))  # CC32 Bank LSB = 0
    track.append(Message('program_change', program=27, time=0))  # GM 28 Electric Clean (0-based 27)


    # Have to work out timing and chords for harmonic notes
    for _ in range(length):
        note = random.randint(50, 70)
        velocity = random.randint(60, 100)
        track.append(Message('note_on', note=note, velocity=velocity, time=0))
        track.append(Message('note_on', note=note, velocity=velocity, time=100))
        track.append(Message('note_on', note=note, velocity=velocity, time=200))
        track.append(Message('note_on', note=note, velocity=velocity, time=300))
        track.append(Message('note_on', note=note, velocity=velocity, time=400))
        track.append(Message('note_on', note=note, velocity=velocity, time=500))
        #track.append(Message('note_off', note=note, velocity=velocity, time=200))

    mid.save(filepath)

# Generate 20 test MIDI files
for i in range(10):
    f = midi_dir / f"riff_{i:03d}.mid"
    create_midi(f)

print("MIDI generation complete")
