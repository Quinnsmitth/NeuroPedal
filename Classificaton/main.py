import os
from chunk import split_audio_file
from train import train_model

RAW_DATA_DIR = "/Volumes/PortableSSD/data/raw" # use file loader to set this path dynamically if needed
SPLIT_DATA_DIR = "/Volumes/PortableSSD/data/split" # use file loader to set this path dynamically if needed
CHUNK_DURATION = 5.0 # seconds
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = .00001

# Preprocess: split audio
os.makedirs(SPLIT_DATA_DIR, exist_ok=True)
for file in os.listdir(RAW_DATA_DIR):
    if file.endswith(".wav"):
        base_name = os.path.splitext(file)[0]
        first_chunk_path = os.path.join(SPLIT_DATA_DIR, f"{base_name}_000.wav")
        if not os.path.exists(first_chunk_path):
            split_audio_file(
                os.path.join(RAW_DATA_DIR, file),
                SPLIT_DATA_DIR,
                CHUNK_DURATION
            )
        else:
            print(f"Skipping {file} (already split)")

# Train the model
train_model(SPLIT_DATA_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)
