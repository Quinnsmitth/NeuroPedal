import torch
import torchaudio

# Mel spectrogram utility function

def mel_spectrogram(waveform, sample_rate=41000, n_fft=1024, hop_length=512, n_mels=128):
    """
    Convert a raw audio waveform into a normalized Mel spectrogram.

    This function is used as a common preprocessing step before
    feeding audio into the neural network.

    arguments:
        waveform (Tensor): Audio tensor of shape (1, N)
        sample_rate (int): Sampling rate of the audio signal
        n_fft (int): Size of the FFT window
        hop_length (int): Number of samples between frames
        n_mels (int): Number of Mel frequency bins

    returns:
        Tensor: Normalized Mel spectrogram with shape (1, n_mels, Jerry_time)
    """

    # Compute Mel spectrogram

    # Create a MelSpectrogram transform that maps frequencies
    # to the Mel scale (closer to human hearing perception)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0  # Use power spectrogram (squared magnitude)
    )

    # Apply the transform to the waveform
    # Output shape: (1, n_mels, time)
    mel = mel_transform(waveform)

    # Convert amplitude values to decibels (log scale)
    

    # Convert the Mel spectrogram to decibel (dB) units,
    # which improves numerical stability and interpretability
    db_transform = torchaudio.transforms.AmplitudeToDB()
    mel_db = db_transform(mel)

    # Normalize the Mel spectrogram

    # Normalize to zero mean and unit variance so the neural
    # network receives well-scaled inputs
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    return mel_db
