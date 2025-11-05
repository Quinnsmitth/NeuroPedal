# src/render_to_wav.py
import subprocess
from pathlib import Path
from select_path import load_config

root = load_config()

midi_dir = root / "midi"
wav_dir = root / "clean"
wav_dir.mkdir(parents=True, exist_ok=True)

project_root = Path(__file__).resolve().parent.parent
soundfont_dir = project_root / "soundfonts"
sf2_files = list(soundfont_dir.glob("*.sf2")) + list(soundfont_dir.glob("*.sf3"))
if not sf2_files:
    raise FileNotFoundError(f"No soundfont found in {soundfont_dir}")

soundfont = str(sf2_files[0])
print(f"🎼 Using soundfont: {soundfont}")

def render_one(midi_path: Path, wav_path: Path):
    # Build an offline, silent render command.
    cmd = [
        "fluidsynth",
        "-ni",
        "-o", "audio.driver=null",  # ✅ silent, offline render
        "-r", "44100",              # sample rate
        "-T", "wav",                # output file type
        "-F", str(wav_path),        # output path
        soundfont,                  # soundfont
        str(midi_path),             # midi file
    ]
    # Run and capture stderr for helpful messages
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Fluidsynth failed.")
        print("Command:", " ".join(cmd))
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        result.check_returncode()  # will raise CalledProcessError

# Optional: quick sanity check for missing MIDIs
midis = list(midi_dir.glob("*.mid"))
if not midis:
    print(f"⚠️ No MIDI files found in {midi_dir}. Run generate_midi.py first.")
else:
    for midi_file in midis:
        wav_file = wav_dir / (midi_file.stem + ".wav")
        render_one(midi_file, wav_file)
        print(f"⚡ Rendered {midi_file.name} → {wav_file.name}")

print(f"✅ Fast WAV rendering complete → {wav_dir}")
