import cv2
import numpy as np
import torch
from pathlib import Path
import dlib
import face_recognition
from typing import List, Tuple, Optional

class VideoProcessor:
    def _init_(self, config):
        self.config = config
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat"
        )
        
    def extract_frames(self, video_path: str, max_frames: int = None) -> np.ndarray:
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
                
        cap.release()
        return np.array(frames)
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        return face_locations
    
    def extract_face_landmarks(self, frame: np.ndarray, face_bbox: Tuple) -> np.ndarray:
        """Extract 68 facial landmarks"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        top, right, bottom, left = face_bbox
        
        rect = dlib.rectangle(left, top, right, bottom)
        landmarks = self.landmark_predictor(gray, rect)
        
        points = []
        for i in range(68):
            points.append([landmarks.part(i).x, landmarks.part(i).y])
            
        return np.array(points)
    
    def crop_face(self, frame: np.ndarray, face_bbox: Tuple, 
                  target_size: Tuple = (299, 299)) -> np.ndarray:
        """Crop and resize face from frame"""
        top, right, bottom, left = face_bbox
        face_crop = frame[top:bottom, left:right]
        face_crop = cv2.resize(face_crop, target_size)
        return face_crop
    
    def process_video(self, video_path: str) -> dict:
        """Process entire video and extract features"""
        frames = self.extract_frames(video_path)
        
        face_crops = []
        landmarks_sequence = []
        
        for frame in frames:
            faces = self.detect_faces(frame)
            
            if faces:
                # Use the largest face
                largest_face = max(faces, key=lambda x: (x[2]-x[0]) * (x[1]-x[3]))
                
                # Crop face
                face_crop = self.crop_face(frame, largest_face)
                face_crops.append(face_crop)
                
                # Extract landmarks
                landmarks = self.extract_face_landmarks(frame, largest_face)
                landmarks_sequence.append(landmarks)
        
        return {
            'face_crops': np.array(face_crops),
            'landmarks': np.array(landmarks_sequence),
            'original_frames': frames
        }