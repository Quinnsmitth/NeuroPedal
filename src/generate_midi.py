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
def create_clean_chord_progression(filepath, tempo_bpm=120, num_bars=4):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(tempo_bpm), time=0))
    track.append(Message('program_change', program=27, time=0))

   # Define basic chords as MIDI note numbers
    C_major = [60, 64, 67]      # C E G
    G_major = [67, 71, 74]      # G B D
    A_minor = [57, 60, 64]      # A C E
    F_major = [65, 69, 72]      # F A C
    D_minor = [62, 65, 69]      # D F A
    E_minor = [64, 67, 71]      # E G B
    B_dim   = [59, 62, 65]      # B D F (diminished)
    D_major = [62, 66, 69] 
# Popular progressions
    chord_progressions = [
        [C_major, G_major, A_minor, F_major],  # I-V-vi-IV
        [A_minor, F_major, C_major, G_major],  # vi-IV-I-V
        [C_major, F_major, G_major, C_major],  # I-IV-V-I
        [D_minor, G_major, C_major, A_minor],  # ii-V-I-vi
        [E_minor, C_major, G_major, D_major],  # Em-C-G-D (common in pop/rock)
        [A_minor, D_minor, G_major, C_major],  # vi-ii-V-I
        [C_major, A_minor, D_minor, G_major],  # I-vi-ii-V
        [F_major, G_major, E_minor, A_minor],  # IV-V-iii-vi
        [G_major, D_major, E_minor, C_major],  # V-I-ii-IV
        [C_major, E_minor, F_major, G_major],  # I-iii-IV-V
    ]

    progression = random.choice(chord_progressions)

    ticks_per_beat = mid.ticks_per_beat
    beats_per_bar = 4
    chord_duration = ticks_per_beat * beats_per_bar

    for chord in progression[:num_bars]:
        for note in chord:
            track.append(Message('note_on', note=note, velocity=90, time=0))
        for note in chord:
            track.append(Message('note_off', note=note, velocity=90, time=chord_duration))

    mid.save(filepath)



metadata = []
for i in range(50):
    tempo = random.choice(range(90, 161, 10))  # 90–160 BPM
    bars = random.choice([1, 2, 4])
    fpath = midi_dir / f"clean_riff_{i:03d}.mid"
    meta = create_clean_guitar_riff(fpath, tempo_bpm=tempo, num_bars=bars)
    metadata.append(meta)

for i in range(50):
    tempo = random.choice(range(90, 161, 10))
    bars = random.choice([1, 2, 4])
    fpath = midi_dir / f"clean_chord_{i:03d}.mid"
    meta = create_clean_chord_progression(fpath, tempo_bpm=tempo, num_bars=bars)
    metadata.append(meta)

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Generated 50 riffs and 50 chord progressions → {midi_dir}")
print(f"Metadata saved to {metadata_path}")