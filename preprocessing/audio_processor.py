import librosa
import numpy as np
import cv2
from typing import Tuple, List

class AudioProcessor:
    def __init__(self, config):
        self.config = config
        self.sr = config['data']['audio_sr']
        
    def extract_audio_from_video(self, video_path: str) -> np.ndarray:
        """Extract audio from video file"""
        import moviepy.editor as mp
        
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        
        # Save temporary audio file
        temp_audio = "temp_audio.wav"
        audio.write_audiofile(temp_audio, verbose=False, logger=None)
        
        # Load with librosa
        y, sr = librosa.load(temp_audio, sr=self.sr)
        
        # Cleanup
        video.close()
        import os
        os.remove(temp_audio)
        
        return y
    
    def extract_mfcc_features(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features from audio"""
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=n_mfcc)
        return mfcc.T  # Transpose for time-first format
    
    def extract_lip_region(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Extract lip region from frame using landmarks"""
        # Lip landmarks are points 48-67
        lip_landmarks = landmarks[48:68]
        
        # Get bounding box of lips
        x_min, y_min = lip_landmarks.min(axis=0)
        x_max, y_max = lip_landmarks.max(axis=0)
        
        # Add padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)
        
        lip_region = frame[y_min:y_max, x_min:x_max]
        return cv2.resize(lip_region, (64, 64))
    
    def synchronize_audio_visual(self, audio_features: np.ndarray, 
                               visual_features: np.ndarray, 
                               fps: float) -> Tuple[np.ndarray, np.ndarray]:
        """Synchronize audio and visual features"""
        # Calculate time alignment
        audio_time_per_frame = len(audio_features) / (len(visual_features) * fps)
        
        # Resample audio features to match video frame rate
        from scipy.interpolate import interp1d
        
        audio_indices = np.linspace(0, len(audio_features) - 1, len(visual_features))
        f = interp1d(np.arange(len(audio_features)), audio_features, 
                    axis=0, kind='linear')
        synchronized_audio = f(audio_indices)
        
        return synchronized_audio, visual_features