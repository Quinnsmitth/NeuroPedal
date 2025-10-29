import os
import torchaudio
from dataset import GuitarPedalDataset
clean_dir = "/Users/quinnsmith/Desktop/guitar_data/clean"
dist_dir = "/Users/quinnsmith/Desktop/guitar_data/dist"

def total_audio_duration(directory):
    total_samples = 0
    total_duration = 0.0  # in seconds

    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):
            path = os.path.join(directory, file_name)
            waveform, sr = torchaudio.load(path)
            num_samples = waveform.shape[-1]
            duration = num_samples / sr

            total_samples += num_samples
            total_duration += duration

    print(f"Total samples: {total_samples:,}")
    print(f"Total duration: {total_duration/60:.2f} minutes ({total_duration:.2f} seconds)")

# Example usage:
total_audio_duration("/Users/quinnsmith/Desktop/guitar_data/clean")
print(GuitarPedalDataset(clean_dir=clean_dir, dist_dir=dist_dir, chunk_size=44100))

