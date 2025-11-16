#src/generate_midi.py

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

# Realistic electric guitar note range (E2–E6)
GUITAR_RANGE = range(40, 89)

# Common guitar scales (intervals from root note)
SCALES = {
    "C_major":      [0, 2, 4, 5, 7, 9, 11, 12],
    "A_minor":      [0, 2, 3, 5, 7, 8, 10, 12],
    "E_minor_pent": [0, 3, 5, 7, 10, 12],
    "A_blues":      [0, 3, 5, 6, 7, 10, 12],
    "G_major":      [0, 2, 4, 5, 7, 9, 11, 12],
    "D_mixolydian": [0, 2, 4, 5, 7, 9, 10, 12],
    "G_mixolydian": [0, 2, 4, 5, 7, 9, 10, 12],
    "C_mixolydian": [0, 2, 4, 5, 7, 9, 10, 12],
    "major_pent": [0, 2, 4, 7, 9, 12],
}

def create_clean_guitar_riff(filepath, tempo_bpm=120, num_bars=2):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    key_name, scale = random.choice(list(SCALES.items()))
    # Select root note within guitar range B2, D3, E3, G3, A3
    root_note = random.choice([47, 50, 52, 55, 57])  # choose mid-neck range

    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(tempo_bpm), time=0))
    track.append(Message('program_change', program=CLEAN_GUITAR_PROGRAM, time=0))

    ticks = mid.ticks_per_beat
    total_length = num_bars * 4 * ticks
    time = 0

    last_note = None

    while time < total_length:
        interval = random.choice(scale)
        note = root_note + interval
        note = max(min(note, max(GUITAR_RANGE)), min(GUITAR_RANGE))

        velocity = random.randint(80, 120)

        duration = random.choice([ticks//2, ticks, ticks*3//2])

        # LEGATO overlap
        overlap = random.randint(15, 45)

        # PICK ATTACK variance
        pick_delay = random.choice([0, 3, 7, -4])

        # HAMMER-ON / PULL-OFF
        if last_note and random.random() < 0.25:
            note = last_note + random.choice([-2, -1, 1, 2])

        # SLIDE (pitch bends)
        if random.random() < 0.15:
            track.append(Message('pitchwheel', pitch=random.choice([-1600, 1600]), time=30))
            track.append(Message('pitchwheel', pitch=0, time=30))

        # NOTE ON
        track.append(Message('note_on', note=note, velocity=velocity, time=max(0, pick_delay)))

        # Vibrato on sustained notes
        if duration > ticks:
            vib = random.randint(100, 600)
            track.append(Message('pitchwheel', pitch=vib, time=duration//3))
            track.append(Message('pitchwheel', pitch=-vib, time=duration//3))
            track.append(Message('pitchwheel', pitch=0, time=duration//3))
        else:
            track.append(Message('note_off', note=note, velocity=velocity, time=duration - overlap))

        time += duration
        last_note = note

    mid.save(filepath)

    return {
        "filename": filepath.name,
        "tempo": tempo_bpm,
        "key": key_name,
        "style": "rock_clean",
        "techniques": ["legato", "vibrato", "hammer_ons", "slides"]
    }

def create_grateful_dead_riff(filepath, tempo_bpm=110, num_bars=4):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Choose key & mode (Mixolydian preferred)
    key_name, scale = random.choice([
        ("D_mixolydian", SCALES["D_mixolydian"]),
        ("G_mixolydian", SCALES["G_mixolydian"]),
        ("C_mixolydian", SCALES["C_mixolydian"]),
        ("major_pent",   SCALES["major_pent"])
    ])

    root_note = random.choice([50, 52, 55, 57])  # mid neck range

    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(tempo_bpm), time=0))
    track.append(Message('program_change', program=CLEAN_GUITAR_PROGRAM, time=0))

    ticks = mid.ticks_per_beat
    total_ticks = num_bars * 4 * ticks
    time = 0

    last_note = None

    while time < total_ticks:
        # choose interval from scale
        interval = random.choice(scale)
        note = root_note + interval

        # humanize velocity for Garcia touch
        velocity = random.randint(70, 105)

        # phrasing: mostly 1/8 notes w/ swing
        duration = random.choice([ticks//2, ticks//2 + ticks//8])

        # add grace slur (hammer-on/pull-off)
        if last_note and random.random() < 0.3:
            note = last_note + random.choice([-2, -1, 1, 2])

        # OCTAVE SLIDE (Jerry trademark)
        if random.random() < 0.12:
            slide_target = note + 12
            track.append(Message("note_on", note=note, velocity=velocity, time=0))
            track.append(Message("pitchwheel", pitch=3000, time=60))
            track.append(Message("note_off", note=note, velocity=velocity, time=10))
            note = slide_target

        # play note
        track.append(Message("note_on", note=note, velocity=velocity, time=max(0, random.randint(-5, 10))))

        # add vibrato to sustained phrases
        if duration > ticks//2 and random.random() < 0.4:
            vib_amt = random.randint(200, 900)
            track.append(Message("pitchwheel", pitch=vib_amt, time=duration//3))
            track.append(Message("pitchwheel", pitch=-vib_amt, time=duration//3))
            track.append(Message("pitchwheel", pitch=0, time=duration//3))
        else:
            track.append(Message("note_off", note=note, velocity=velocity, time=duration))

        last_note = note
        time += duration

    mid.save(filepath)

    return {
        "filename": filepath.name,
        "style": "grateful_dead_lead",
        "key": key_name,
        "techniques": ["mixolydian", "grace_notes", "vibrato", "slides", "pentatonic coloration"]
    }

def create_clean_chord_progression(filepath, tempo_bpm=120, num_bars=4):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(tempo_bpm), time=0))
    track.append(Message('program_change', program=CLEAN_GUITAR_PROGRAM, time=0))

    chords = [
        [60, 64, 67], [67, 71, 74], [57, 60, 64], [65, 69, 72],
        [62, 65, 69], [64, 67, 71], [59, 62, 65]
    ]
    progression = random.choices(chords, k=num_bars)

    ticks = mid.ticks_per_beat
    chord_dur = 4 * ticks

    for chord in progression:
        # STRUM — downward pick (low strings first)
        for i, note in enumerate(chord):
            track.append(Message('note_on', note=note, velocity=90 - i*8, time=i * 25))

        # Light vibrato across chord
        track.append(Message('pitchwheel', pitch=400, time=chord_dur//3))
        track.append(Message('pitchwheel', pitch=-400, time=chord_dur//3))
        track.append(Message('pitchwheel', pitch=0, time=chord_dur//3))

        # NOTE OFF
        for note in chord:
            track.append(Message('note_off', note=note, velocity=64, time=3))

        # GHOST CHORD TAP (muted strum)
        if random.random() < 0.3:
            ghost = chord[random.randint(0, len(chord)-1)]
            track.append(Message('note_on', note=ghost, velocity=30, time=20))
            track.append(Message('note_off', note=ghost, velocity=30, time=40))

    mid.save(filepath)



metadata = []
for i in range(10):
    tempo = random.choice(range(90, 161, 10))  # 90–160 BPM
    bars = random.choice([1, 2, 4])
    fpath = midi_dir / f"clean_riff_{i:03d}.mid"
    meta = create_clean_guitar_riff(fpath, tempo_bpm=tempo, num_bars=bars)
    metadata.append(meta)

for i in range(10):
    tempo = random.choice(range(90, 140, 5))  # Dead groove tempos
    bars = random.choice([2, 4, 8])
    fpath = midi_dir / f"dead_riff_{i:03d}.mid"
    meta = create_grateful_dead_riff(fpath, tempo_bpm=tempo, num_bars=bars)
    metadata.append(meta)

for i in range(10):
    tempo = random.choice(range(90, 161, 10))
    bars = random.choice([1, 2, 4])
    fpath = midi_dir / f"clean_chord_{i:03d}.mid"
    meta = create_clean_chord_progression(fpath, tempo_bpm=tempo, num_bars=bars)
    metadata.append(meta)

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Generated 50 riffs and 50 chord progressions -> {midi_dir}")
print(f"Metadata saved to {metadata_path}")