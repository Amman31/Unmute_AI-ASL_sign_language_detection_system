"""
Hand detection module using MediaPipe.
Extracts hand landmarks from images and video frames.
"""
import mediapipe as mp
import cv2
import numpy as np
from typing import Optional, Tuple, List

from .config import (
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_STATIC_IMAGE_MODE
)


class HandDetector:
    """Hand detection and landmark extraction using MediaPipe."""
    
    def __init__(self):
        """Initialize MediaPipe hands detector."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=MEDIAPIPE_STATIC_IMAGE_MODE,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[List[float]]:
        """
        Extract normalized hand landmarks from an image.
        
        Args:
            image: Input image in RGB format
            
        Returns:
            List of normalized landmark coordinates (42 values: 21 landmarks * 2 coords)
            or None if no hand detected
        """
        results = self.hands.process(image)
        
        if not results.multi_hand_landmarks:
            return None
        
        data_aux = []
        x_coords = []
        y_coords = []
        
        # Process first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Collect all coordinates
        for landmark in hand_landmarks.landmark:
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)
        
        # Normalize coordinates relative to minimum values
        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x - min(x_coords))
            data_aux.append(landmark.y - min(y_coords))
        
        # Return 42 values (21 landmarks * 2 coordinates)
        if len(data_aux) == 42:
            return data_aux
        
        return None
    
    def process_frame(
        self, 
        frame: np.ndarray
    ) -> Tuple[np.ndarray, Optional[List[float]], Optional[Tuple[int, int, int, int]]]:
        """
        Process a video frame and extract hand landmarks with bounding box.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Tuple of (annotated_frame, landmarks, bounding_box)
            - annotated_frame: Frame with hand landmarks drawn
            - landmarks: List of normalized landmark coordinates or None
            - bounding_box: (x1, y1, x2, y2) or None
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        annotated_frame = frame.copy()
        landmarks = None
        bbox = None
        
        if results.multi_hand_landmarks:
            data_aux = []
            x_coords = []
            y_coords = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Collect coordinates
                for landmark in hand_landmarks.landmark:
                    x_coords.append(landmark.x)
                    y_coords.append(landmark.y)
                
                # Normalize coordinates
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_coords))
                    data_aux.append(landmark.y - min(y_coords))
            
            # Calculate bounding box
            H, W, _ = frame.shape
            x1 = int(min(x_coords) * W) - 10
            y1 = int(min(y_coords) * H) - 10
            x2 = int(max(x_coords) * W) + 10
            y2 = int(max(y_coords) * H) + 10
            
            bbox = (x1, y1, x2, y2)
            
            if len(data_aux) == 42:
                landmarks = data_aux
        
        return annotated_frame, landmarks, bbox
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'hands'):
            self.hands.close()

