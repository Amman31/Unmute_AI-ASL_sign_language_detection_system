"""
Configuration module for UNMUTE AI application.
Contains all constants, paths, and configuration settings.
"""
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "image_dataset"
OUTPUT_DIR = BASE_DIR / "output"

# Output files
DATA_PICKLE_FILE = OUTPUT_DIR / "data.pickle"
MODEL_FILE = OUTPUT_DIR / "model.p"

# MediaPipe settings
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.3
MEDIAPIPE_STATIC_IMAGE_MODE = True

# Model settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Application settings
DETECTION_TIMER_THRESHOLD = 30  # frames
SIGN_REFERENCE_IMAGE = BASE_DIR / "signs.png"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

