"""
Data processing module for creating dataset from images.
Processes sign language images and extracts MediaPipe hand landmarks.
"""
import os
import pickle
import cv2
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple, Optional

from .config import DATA_DIR, DATA_PICKLE_FILE
from .hand_detector import HandDetector


def process_image(args: Tuple[str, str]) -> Optional[Tuple[List[float], str]]:
    """
    Process a single image and extract hand landmarks.
    
    Args:
        args: Tuple of (directory_name, image_filename)
        
    Returns:
        Tuple of (landmarks, label) or None if processing failed
    """
    dir_name, img_filename = args
    detector = HandDetector()
    
    img_path = Path(DATA_DIR) / dir_name / img_filename
    if not img_path.exists():
        return None
    
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    landmarks = detector.extract_landmarks(img_rgb)
    
    if landmarks:
        return landmarks, dir_name
    
    return None


def create_dataset() -> None:
    """
    Create dataset from images in the data directory.
    Processes all images and saves extracted landmarks to a pickle file.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}\n"
            f"Please ensure the image_dataset folder exists with sign language images."
        )
    
    print(f"Processing images from {DATA_DIR}...")
    
    # Collect all image paths
    img_paths = []
    for dir_name in os.listdir(DATA_DIR):
        dir_path = DATA_DIR / dir_name
        if dir_path.is_dir():
            for img_filename in os.listdir(dir_path):
                img_paths.append((dir_name, img_filename))
    
    if not img_paths:
        raise ValueError(f"No images found in {DATA_DIR}")
    
    print(f"Found {len(img_paths)} images to process...")
    
    # Process images in parallel
    data = []
    labels = []
    
    with Pool() as pool:
        results = pool.map(process_image, img_paths)
    
    # Collect valid results
    for result in results:
        if result:
            landmarks, label = result
            data.append(landmarks)
            labels.append(label)
    
    if not data:
        raise ValueError("No valid hand landmarks extracted from images. Check your images.")
    
    print(f"Successfully processed {len(data)} images.")
    print(f"Labels found: {set(labels)}")
    
    # Save to pickle file
    with open(DATA_PICKLE_FILE, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    
    print(f"Dataset saved to {DATA_PICKLE_FILE}")

