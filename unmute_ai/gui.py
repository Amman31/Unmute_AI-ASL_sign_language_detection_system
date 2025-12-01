"""
GUI module for the UNMUTE AI application.
Provides the main Tkinter interface for real-time sign language detection.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import pygame
from gtts import gTTS
from io import BytesIO
from pathlib import Path

from .config import MODEL_FILE, DETECTION_TIMER_THRESHOLD, SIGN_REFERENCE_IMAGE
from .hand_detector import HandDetector


class SignLanguageApp:
    """Main application class for sign language detection GUI."""
    
    def __init__(self):
        """Initialize the application."""
        self.model = self._load_model()
        self.detector = HandDetector()
        self.cap = None
        self.root = None
        
        # State variables
        self.current_char = None
        self.timer = 0
        self.sentence = ''
        self.spoken_sentences = []
        
        # Initialize pygame for audio
        pygame.mixer.init()
    
    def _load_model(self):
        """Load the trained model from file."""
        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_FILE}\n"
                f"Please run train.py first to generate the model."
            )
        
        import pickle
        with open(MODEL_FILE, 'rb') as f:
            model_dict = pickle.load(f)
        return model_dict['model']
    
    def _setup_ui(self) -> None:
        """Set up the Tkinter user interface."""
        self.root = tk.Tk()
        self.root.title("UNMUTE AI")
        self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Configure styles
        style = ttk.Style()
        style.configure("TFrame", background="lightblue")
        style.configure("TLabel", background="lightblue", font=("Arial", 18))
        style.configure("TProgressbar", thickness=20)
        style.configure("delete.TButton", background="red", foreground="white")
        style.configure("clear.TButton", background="green", foreground="white")
        style.configure("view.TButton", background="blue", foreground="white")
        
        # Main layout
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left frame
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Buttons frame
        button_frame = ttk.Frame(left_frame, width=300)
        button_frame.pack(fill=tk.BOTH, pady=5)
        
        btn_delete = ttk.Button(
            button_frame, 
            text="DELETE", 
            style="delete.TButton", 
            command=self._delete_last_character
        )
        btn_delete.pack(side=tk.LEFT, padx=20, pady=20, expand=True)
        
        btn_clear = ttk.Button(
            button_frame, 
            text="CLEAR", 
            style="clear.TButton",
            command=self._clear_text
        )
        btn_clear.pack(side=tk.LEFT, padx=20, pady=20, expand=True)
        
        btn_view_signs = ttk.Button(
            button_frame, 
            text="VIEW SIGNS", 
            style="view.TButton", 
            command=self._view_signs
        )
        btn_view_signs.pack(side=tk.LEFT, padx=20, pady=20, expand=True)
        
        # Right frame
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Text output frame
        text_frame = ttk.Frame(left_frame, width=300)
        text_frame.pack(fill=tk.BOTH, pady=5)
        
        ttk.Label(text_frame, text="Real-Time Text").pack(anchor=tk.W)
        self.text_output = tk.Text(text_frame, height=5, width=40, font=("Arial", 18))
        self.text_output.pack(pady=5, fill=tk.BOTH, expand=False)
        
        self.progress = ttk.Progressbar(text_frame, mode='determinate')
        self.progress.pack(pady=5, fill=tk.X)
        
        # Camera feed frame
        camera_frame = ttk.Frame(right_frame)
        camera_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Label(camera_frame, text="Camera Feed").pack(anchor=tk.W)
        self.video_label = tk.Label(camera_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Spoken sentences frame
        sentences_frame = ttk.Frame(left_frame, width=300)
        sentences_frame.pack(fill=tk.BOTH, pady=5)
        
        ttk.Label(sentences_frame, text="Spoken Sentences").pack(anchor=tk.W)
        self.sentences_box = tk.Text(sentences_frame, font=("Arial", 18), width=40)
        self.sentences_box.pack(pady=5, fill=tk.BOTH, expand=False)
    
    def _delete_last_character(self) -> None:
        """Delete the last character from the current sentence."""
        if self.sentence:
            self.sentence = self.sentence[:-1]
    
    def _clear_text(self) -> None:
        """Clear the current sentence."""
        self.sentence = ""
    
    def _view_signs(self) -> None:
        """Open a window displaying sign language reference image."""
        if not SIGN_REFERENCE_IMAGE.exists():
            messagebox.showwarning(
                "File Not Found",
                f"Sign reference image not found: {SIGN_REFERENCE_IMAGE}"
            )
            return
        
        new_window = tk.Toplevel(self.root)
        new_window.title("Sign Language Reference")
        new_window.geometry("1280x720")
        
        img = Image.open(SIGN_REFERENCE_IMAGE)
        img = img.resize((1280, 720), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        
        img_label = tk.Label(new_window, image=img_tk)
        img_label.image = img_tk  # Keep reference
        img_label.pack(fill=tk.BOTH, expand=True)
    
    def _speak_text(self, text: str) -> None:
        """
        Convert text to speech and play it.
        
        Args:
            text: Text to speak
        """
        try:
            audio_buffer = BytesIO()
            speech = gTTS(text=text, lang='en', slow=False)
            speech.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            pygame.mixer.music.load(audio_buffer, 'mp3')
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
    
    def _update_ui(self) -> None:
        """Update the UI with the latest frame and detection results."""
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self._update_ui)
            return
        
        # Process frame
        annotated_frame, landmarks, bbox = self.detector.process_frame(frame)
        
        # Predict character
        detected_char = None
        if landmarks:
            try:
                prediction = self.model.predict([np.asarray(landmarks)])
                detected_char = prediction[0]
            except ValueError:
                pass
        
        # Update timer and sentence
        if detected_char == self.current_char:
            self.timer += 1
        else:
            self.timer = 0
            self.current_char = detected_char
        
        # Handle character detection
        if self.timer >= DETECTION_TIMER_THRESHOLD and self.current_char:
            if self.current_char == 'space':
                self.sentence += ' '
                self._speak_text('space')
            elif self.current_char == 'del':
                if self.sentence:
                    self.spoken_sentences.append(self.sentence)
                    self.sentences_box.delete(1.0, tk.END)
                    self.sentences_box.insert(tk.END, "\n".join(self.spoken_sentences))
                    self._speak_text(self.sentence)
                    self.sentence = ''
            else:
                self.sentence += self.current_char
                self._speak_text(self.current_char)
            
            self.timer = 0
        
        # Draw bounding box and label
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            if detected_char:
                label = 'ENTER' if detected_char == 'del' else detected_char
                cv2.putText(
                    annotated_frame, 
                    label, 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.3, 
                    (0, 0, 255), 
                    3, 
                    cv2.LINE_AA
                )
        
        # Update text output
        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.END, self.sentence)
        
        # Update progress bar
        self.progress['value'] = (self.timer / DETECTION_TIMER_THRESHOLD) * 100
        
        # Update video display
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        screen_width = self.root.winfo_screenwidth()
        aspect_ratio = frame_rgb.shape[1] / frame_rgb.shape[0]
        new_width = int(screen_width * 0.6)
        new_height = int(new_width / aspect_ratio)
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
        
        img = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
        self.video_label.config(image=img)
        self.video_label.image = img
        
        self.root.after(10, self._update_ui)
    
    def _on_closing(self) -> None:
        """Handle window closing event."""
        if self.cap:
            self.cap.release()
        self.root.destroy()
    
    def run(self) -> None:
        """Run the application."""
        # Open camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera. Please ensure a camera is connected.")
        
        # Setup UI
        self._setup_ui()
        
        # Start update loop
        self._update_ui()
        
        # Run main loop
        self.root.mainloop()

