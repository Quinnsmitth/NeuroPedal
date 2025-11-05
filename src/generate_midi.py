# src/generate_midi.py

from mido import MidiFile, MidiTrack, Message
from pathlib import Path
import random
from select_path import load_config

# Load correct dataset root
root = load_config()

midi_dir = root / "midi"
midi_dir.mkdir(parents=True, exist_ok=True)

print(f"🎼 Writing MIDI files to: {midi_dir}")

def create_midi(filepath, program=24, length=8):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Set guitar instrument (General MIDI Patch 24 = Nylon Guitar)
    track.append(Message('program_change', program=program, time=0))

    for _ in range(length):
        note = random.randint(50, 70)
        velocity = random.randint(60, 100)
        track.append(Message('note_on', note=note, velocity=velocity, time=0))
        track.append(Message('note_off', note=note, velocity=velocity, time=480))

    mid.save(filepath)

# Generate 20 test MIDI files
for i in range(20):
    f = midi_dir / f"riff_{i:03d}.mid"
    create_midi(f)

print("✅ MIDI generation complete")
