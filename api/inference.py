import torch
import cv2
import numpy as np
from pathlib import Path
import tempfile
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.ensemble_model import DeepFakeEnsemble
from preprocessing.video_processor import VideoProcessor
from preprocessing.audio_processor import AudioProcessor
import yaml

class DeepFakeDetector:
    def __init__(self, model_path, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = DeepFakeEnsemble(self.config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize processors
        self.video_processor = VideoProcessor(self.config)
        self.audio_processor = AudioProcessor(self.config)
        
    def preprocess_video(self, video_path):
        """Preprocess video for inference"""
        try:
            # Process video
            video_data = self.video_processor.process_video(video_path)
            face_crops = video_data['face_crops']
            landmarks = video_data['landmarks']
            
            if len(face_crops) == 0:
                raise ValueError("No faces detected in video")
            
            # Process audio
            audio = self.audio_processor.extract_audio_from_video(video_path)
            audio_features = self.audio_processor.extract_mfcc_features(audio)
            
            # Extract lip regions
            lip_regions = []
            for i, (frame, landmark) in enumerate(zip(face_crops, landmarks)):
                lip_region = self.audio_processor.extract_lip_region(frame, landmark)
                lip_regions.append(lip_region)
            
            # Convert to tensors
            face_crops = torch.FloatTensor(face_crops).permute(0, 3, 1, 2) / 255.0
            landmarks = torch.FloatTensor(landmarks)
            lip_regions = torch.FloatTensor(lip_regions).permute(0, 3, 1, 2) / 255.0
            
            # Create sequences
            sequence_length = min(16, len(face_crops))  # Use available frames or max 16
            
            # Sample sequence from middle of video
            if len(face_crops) >= sequence_length:
                start_idx = (len(face_crops) - sequence_length) // 2
                face_sequence = face_crops[start_idx:start_idx + sequence_length]
                landmark_sequence = landmarks[start_idx:start_idx + sequence_length]
                lip_sequence = lip_regions[start_idx:start_idx + sequence_length]
            else:
                # Pad sequence
                pad_length = sequence_length - len(face_crops)
                face_sequence = torch.cat([
                    face_crops,
                    face_crops[-1:].repeat(pad_length, 1, 1, 1)
                ])
                landmark_sequence = torch.cat([
                    landmarks,
                    landmarks[-1:].repeat(pad_length, 1, 1)
                ])
                lip_sequence = torch.cat([
                    lip_regions,
                    lip_regions[-1:].repeat(pad_length, 1, 1, 1)
                ])
            
            # Resample audio to match sequence
            if len(audio_features) != sequence_length:
                from scipy.interpolate import interp1d
                if len(audio_features) > 1:
                    f = interp1d(np.arange(len(audio_features)), audio_features, 
                                axis=0, kind='linear')
                    audio_indices = np.linspace(0, len(audio_features) - 1, sequence_length)
                    audio_sequence = torch.FloatTensor(f(audio_indices))
                else:
                    audio_sequence = torch.FloatTensor(audio_features).repeat(sequence_length, 1)
            else:
                audio_sequence = torch.FloatTensor(audio_features)
            
            return {
                'frames': face_sequence[0].unsqueeze(0),  # Add batch dimension
                'sequences': face_sequence.mean(dim=[2, 3]).unsqueeze(0),
                'landmarks': landmark_sequence.unsqueeze(0),
                'audio': audio_sequence.unsqueeze(0),
                'lip_regions': lip_sequence[0].unsqueeze(0)
            }
            
        except Exception as e:
            raise RuntimeError(f"Error preprocessing video: {str(e)}")
    
    def predict(self, video_path):
        """Predict if video is deepfake"""
        try:
            # Preprocess video
            batch_data = self.preprocess_video(video_path)
            
            # Move to device
            for key in batch_data:
                batch_data[key] = batch_data[key].to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(batch_data)
                
                # Get probabilities
                final_probs = torch.softmax(outputs['final_output'], dim=1)
                spatial_probs = torch.softmax(outputs['spatial_output'], dim=1)
                temporal_probs = torch.softmax(outputs['temporal_output'], dim=1)
                landmark_probs = torch.softmax(outputs['landmark_output'], dim=1)
                sync_probs = torch.softmax(outputs['sync_output'], dim=1)
                
                # Get predictions
                final_pred = torch.argmax(final_probs, dim=1)
                confidence = torch.max(final_probs, dim=1)[0]
                
                return {
                    'prediction': 'fake' if final_pred.item() == 1 else 'real',
                    'confidence': confidence.item(),
                    'fake_probability': final_probs[0, 1].item(),
                    'real_probability': final_probs[0, 0].item(),
                    'component_predictions': {
                        'spatial': {
                            'fake_prob': spatial_probs[0, 1].item(),
                            'real_prob': spatial_probs[0, 0].item()
                        },
                        'temporal': {
                            'fake_prob': temporal_probs[0, 1].item(),
                            'real_prob': temporal_probs[0, 0].item()
                        },
                        'landmarks': {
                            'fake_prob': landmark_probs[0, 1].item(),
                            'real_prob': landmark_probs[0, 0].item()
                        },
                        'audio_visual_sync': {
                            'fake_prob': sync_probs[0, 1].item(),
                            'real_prob': sync_probs[0, 0].item()
                        }
                    }
                }
                
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")
