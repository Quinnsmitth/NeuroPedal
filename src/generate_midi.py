# src/generate_midi.py
import mido, random
from mido import MidiFile, MidiTrack, Message
from pathlib import Path

def create_midi(output_path, program=24, num_notes=8):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(Message('program_change', program=program, time=0))

    for _ in range(num_notes):
        note = random.randint(50, 70)
        vel = random.randint(60, 100)
        track.append(Message('note_on', note=note, velocity=vel, time=0))
        track.append(Message('note_off', note=note, velocity=vel, time=480))
    mid.save(output_path)

if __name__ == "__main__":
    Path("data/midi/clean").mkdir(parents=True, exist_ok=True)
    for i in range(100):
        create_midi(f"data/midi/clean/clean_{i}.mid")
