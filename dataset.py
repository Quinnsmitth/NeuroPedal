import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio

def safe_normalize(tensor):
    max_val = torch.max(torch.abs(tensor))
    return tensor / max_val if max_val > 0 else tensor

class GuitarPedalDataset(Dataset):
    def __init__(self, clean_dir, dist_dir, chunk_size=44100):
        self.clean_dir = clean_dir
        self.dist_dir = dist_dir
        self.chunk_size = chunk_size

        self.clean_files = sorted(os.listdir(clean_dir))
        self.dist_files = sorted(os.listdir(dist_dir))

        # Precompute the chunk indices for all files
        self.chunks = []  # list of tuples: (file_idx, start_idx, end_idx)
        for idx, (clean_file, dist_file) in enumerate(zip(self.clean_files, self.dist_files)):
            clean_path = os.path.join(clean_dir, clean_file)
            dist_path = os.path.join(dist_dir, dist_file)

            y_clean, _ = torchaudio.load(clean_path)
            y_dist, _ = torchaudio.load(dist_path)

            min_len = min(y_clean.shape[-1], y_dist.shape[-1])
            num_chunks = (min_len + chunk_size - 1) // chunk_size  # ceil division

            for i in range(num_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, min_len)
                self.chunks.append((idx, start, end))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        file_idx, start, end = self.chunks[idx]
        clean_path = os.path.join(self.clean_dir, self.clean_files[file_idx])
        dist_path = os.path.join(self.dist_dir, self.dist_files[file_idx])

        y_clean, _ = torchaudio.load(clean_path)
        y_dist, _ = torchaudio.load(dist_path)

        # Convert to mono if stereo
        if y_clean.shape[0] > 1:
            y_clean = y_clean.mean(dim=0, keepdim=True)
        if y_dist.shape[0] > 1:
            y_dist = y_dist.mean(dim=0, keepdim=True)

        # Match lengths
        min_len = min(y_clean.shape[-1], y_dist.shape[-1])
        y_clean = y_clean[:, :min_len]
        y_dist = y_dist[:, :min_len]

        # Extract the chunk
        y_clean = y_clean[:, start:end]
        y_dist = y_dist[:, start:end]

        # Pad if chunk is shorter than chunk_size (last chunk of file)
        pad_len = self.chunk_size - y_clean.shape[-1]
        if pad_len > 0:
            y_clean = F.pad(y_clean, (0, pad_len))
            y_dist = F.pad(y_dist, (0, pad_len))

        # Normalize
        y_clean = safe_normalize(y_clean)
        y_dist = safe_normalize(y_dist)

        # Return only tensors, not sample rate
        return y_clean, y_dist
