import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import json
import librosa
from preprocessing.video_processor import VideoProcessor
from preprocessing.audio_processor import AudioProcessor

class DeepFakeDataset(Dataset):
    def __init__(self, data_path, config, split='train', transform=None):
        self.data_path = Path(data_path)
        self.config = config
        self.split = split
        self.transform = transform
        
        # Initialize processors
        self.video_processor = VideoProcessor(config)
        self.audio_processor = AudioProcessor(config)
        
        # Load dataset annotations
        self.annotations = self.load_annotations()
        
        # Filter by split
        self.samples = [s for s in self.annotations if s['split'] == split]
        
    def load_annotations(self):
        """Load dataset annotations"""
        annotations_file = self.data_path / 'annotations.json'
        
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                return json.load(f)
        else:
            # Create annotations from directory structure
            return self.create_annotations_from_structure()
    
    def create_annotations_from_structure(self):
        """Create annotations from directory structure"""
        annotations = []
        
        # Assume structure: data/real/ and data/fake/
        for label, folder in [('real', 'real'), ('fake', 'fake')]:
            folder_path = self.data_path / folder
            if folder_path.exists():
                for video_file in folder_path.glob('*.mp4'):
                    annotations.append({
                        'video_path': str(video_file),
                        'label': 0 if label == 'real' else 1,
                        'split': 'train'  # You'll need to implement train/val splitting
                    })
                    
        return annotations
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['video_path']
        label = sample['label']
        
        try:
            # Process video
            video_data = self.video_processor.process_video(video_path)
            face_crops = video_data['face_crops']
            landmarks = video_data['landmarks']
            
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
            audio_features = torch.FloatTensor(audio_features)
            lip_regions = torch.FloatTensor(lip_regions).permute(0, 3, 1, 2) / 255.0
            
            # Create sequences for temporal analysis
            sequence_length = self.config['model']['temporal']['sequence_length']
            if len(face_crops) >= sequence_length:
                # Sample random sequence
                start_idx = np.random.randint(0, len(face_crops) - sequence_length + 1)
                face_sequence = face_crops[start_idx:start_idx + sequence_length]
                landmark_sequence = landmarks[start_idx:start_idx + sequence_length]
                lip_sequence = lip_regions[start_idx:start_idx + sequence_length]
                
                # Adjust audio features to match sequence
                audio_start = int(start_idx * len(audio_features) / len(face_crops))
                audio_end = int((start_idx + sequence_length) * len(audio_features) / len(face_crops))
                audio_sequence = audio_features[audio_start:audio_end]
                
                # Resample audio to match sequence length
                if len(audio_sequence) != sequence_length:
                    from scipy.interpolate import interp1d
                    f = interp1d(np.arange(len(audio_sequence)), audio_sequence, 
                                axis=0, kind='linear')
                    audio_indices = np.linspace(0, len(audio_sequence) - 1, sequence_length)
                    audio_sequence = torch.FloatTensor(f(audio_indices))
                
            else:
                # Pad if sequence is too short
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
                
                # Pad audio sequence
                audio_pad = sequence_length - len(audio_features)
                if audio_pad > 0:
                    audio_sequence = torch.cat([
                        audio_features,
                        audio_features[-1:].repeat(audio_pad, 1)
                    ])
                else:
                    audio_sequence = audio_features[:sequence_length]
                audio_sequence = torch.FloatTensor(audio_sequence)
            
            batch_data = {
                'frames': face_sequence[0],  # Single frame for spatial analysis
                'sequences': face_sequence.mean(dim=[2, 3]),  # Pooled features for temporal
                'landmarks': landmark_sequence,
                'audio': audio_sequence,
                'lip_regions': lip_sequence[0]  # Single lip frame for sync analysis
            }
            
            return batch_data, torch.LongTensor([label])
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            # Return a dummy sample
            return self.get_dummy_sample(), torch.LongTensor([0])
    
    def get_dummy_sample(self):
        """Return a dummy sample for error cases"""
        batch_data = {
            'frames': torch.zeros(3, 299, 299),
            'sequences': torch.zeros(16, 2048),  # Assuming feature size
            'landmarks': torch.zeros(16, 68, 2),
            'audio': torch.zeros(16, 13),
            'lip_regions': torch.zeros(3, 64, 64)
        }
        return batch_data
