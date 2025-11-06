from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
from pathlib import Path
import random, json
from select_path import load_config


root = load_config()
midi_dir = root / "midi"
metadata_path = root / "metadata" / "riffs_metadata_clean.json"

midi_dir.mkdir(parents=True, exist_ok=True)
metadata_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Writing clean guitar riffs to: {midi_dir}")


# Electric guitar (clean) program number (General MIDI 28 → index 27)
CLEAN_GUITAR_PROGRAM = 27

# Realistic electric guitar note range (E2–A5)
GUITAR_RANGE = range(40, 89)

# Common guitar scales (intervals from root note)
SCALES = {
    "C_major":      [0, 2, 4, 5, 7, 9, 11, 12],
    "A_minor":      [0, 2, 3, 5, 7, 8, 10, 12],
    "E_minor_pent": [0, 3, 5, 7, 10, 12],
    "A_blues":      [0, 3, 5, 6, 7, 10, 12],
    "G_major":      [0, 2, 4, 5, 7, 9, 11, 12],
}

def create_clean_guitar_riff(filepath, tempo_bpm=120, num_bars=2):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Choose scale and key
    key_name, scale = random.choice(list(SCALES.items()))
    root_note = random.choice([40, 45, 50, 52, 55, 57, 59])  # E2–B3

    # Tempo metadata
    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(tempo_bpm), time=0))

    # Set instrument to clean electric guitar
    track.append(Message('program_change', program=CLEAN_GUITAR_PROGRAM, time=0))

    # MIDI timing setup
    ticks_per_beat = mid.ticks_per_beat
    riff_length_ticks = num_bars * 4 * ticks_per_beat  # 4 beats per bar
    time = 0

    while time < riff_length_ticks:
        # Pick a note from the scale
        interval = random.choice(scale)
        note = root_note + interval
        if note not in GUITAR_RANGE:
            note = min(max(note, min(GUITAR_RANGE)), max(GUITAR_RANGE))

        # Duration: 1/8, 1/16, or 1/4 note
        duration = random.choice([ticks_per_beat // 2, ticks_per_beat, ticks_per_beat * 3 // 2])

        # Small human timing variation
        delay = random.choice([0, 10, 20, -10])

        # Velocity for pluck strength
        velocity = random.randint(70, 115)
        if random.random() < 0.1:
            # occasional ghost note
            velocity = random.randint(40, 60)

        # Add the note events
        track.append(Message('note_on', note=note, velocity=velocity, time=max(0, delay)))
        track.append(Message('note_off', note=note, velocity=velocity, time=duration))

        time += duration

    mid.save(filepath)

    # Return metadata for this riff
    return {
        "filename": filepath.name,
        "tempo": tempo_bpm,
        "key": key_name,
        "program": CLEAN_GUITAR_PROGRAM,
        "bars": num_bars,
        "instrument": "clean_electric_guitar"
    }

metadata = []
num_files = 100

for i in range(num_files):
    tempo = random.choice(range(90, 161, 10))  # 90–160 BPM
    bars = random.choice([1, 2, 4])
    fpath = midi_dir / f"clean_riff_{i:03d}.mid"
    meta = create_clean_guitar_riff(fpath, tempo_bpm=tempo, num_bars=bars)
    metadata.append(meta)

# Save metadata file
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"\n Generated {num_files} clean guitar riffs and saved metadata to {metadata_path}")



## def create_clean_chord_progression(filepath, tempo_bpm=120, num_bars=4):
##     mid = MidiFile()
##     track = MidiTrack()