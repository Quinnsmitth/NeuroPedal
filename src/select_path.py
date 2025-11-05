# src/select_path.py
"""
-----------------------------------------
Detects or asks for the user's data root directory
(e.g., USB, external SSD, or local folder).

USB Folder Setup:
USB\
    guitar_data\
        clean\
        distorted\
        midi\
        features\

Creates a config.json file in the project root so
other scripts can automatically find the correct base path.
-----------------------------------------
"""

import json
from pathlib import Path
import platform

CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"


def detect_default_paths():
    """Return a list of likely data locations depending on OS."""
    system = platform.system()
    guesses = []

    if system == "Windows":
        for drive in "DEFGHIJKLMNOPQRSTUVWXYZ":
            guesses.append(Path(f"{drive}:/guitar_data"))
    elif system == "Darwin":  # macOS
        guesses += [          
            Path("/Volumes/PortableSSD/guitar_data"),      # common SSD setup
            Path.home() / "Desktop/guitar_data"    # fallback local path
        ]
    else:  # Linux or others
        guesses += [
            Path("/media") / Path.home().name / "guitar_data",
            Path.home() / "guitar_data"
        ]
    return guesses


def select_data_root():
    """Find or ask for the base data folder."""
    print("\nSearching for guitar_data folder...")

    for candidate in detect_default_paths():
        if candidate.exists():
            print(f"Found existing folder: {candidate}")
            return candidate

    # If not found, ask the user
    print("⚠️ Could not automatically find your data folder.")
    manual = input("Please drag your 'guitar_data' folder here or type its path: ").strip()
    path = Path(manual).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return path


def save_config(data_root: Path):
    """Write the chosen data path to config.json."""
    config = {"data_root": str(data_root)}
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved data path to {CONFIG_FILE}")


def load_config() -> Path:
    """Load saved config; prompt if missing."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
            return Path(data["data_root"])
    else:
        root = select_data_root()
        save_config(root)
        return root


if __name__ == "__main__":
    root = select_data_root()
    save_config(root)
    print(f"Using data root: {root}")
