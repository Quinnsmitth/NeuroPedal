from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
import random

# Create a new MIDI file and track
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

# Meta information
track.append(MetaMessage('set_tempo', tempo=bpm2tempo(100)))  # 100 BPM
track.append(MetaMessage('track_name', name='Simple Guitar Riff'))

# General MIDI program number for clean electric guitar (program 27)
track.append(Message('program_change', program=27, time=0))

# Define a simple C major riff (MIDI note numbers for E2–A5 range)
notes = [48, 52, 55, 60, 64, 67]  # C major chord tones

# Generate 8 random strums
for i in range(8):
    note = random.choice(notes)
    velocity = random.randint(70, 110)
    duration = random.choice([240, 480])  # quarter or half note

    # Note on / note off events
    track.append(Message('note_on', note=note, velocity=velocity, time=0))
    track.append(Message('note_off', note=note, velocity=0, time=duration))

# Save to file
mid.save('simple_guitar_riff.mid')
print("✅ Saved MIDI file: simple_guitar_riff.mid")
