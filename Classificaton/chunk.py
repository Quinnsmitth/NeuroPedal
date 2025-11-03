import os
import math
import torchaudio

def split_audio_file(input_path, output_dir, chunk_duration=5.0):
    #Split a .wav file into N chunks of given duration (seconds).
    os.makedirs(output_dir, exist_ok=True)
    waveform, sr = torchaudio.load(input_path)
    total_duration = waveform.size(1) / sr
    num_chunks = math.ceil(total_duration / chunk_duration)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    print(f"Splitting {base_name}: {num_chunks} chunks of {chunk_duration}s")

    for i in range(num_chunks):
        start = int(i * chunk_duration * sr)
        end = min(int((i + 1) * chunk_duration * sr), waveform.size(1))
        chunk = waveform[:, start:end]
        if chunk.size(1) == 0:
            continue
        output_path = os.path.join(output_dir, f"{base_name}_{i:03d}.wav")
        torchaudio.save(output_path, chunk, sr)

    print(f"Done: saved {num_chunks} chunks to {output_dir}")
