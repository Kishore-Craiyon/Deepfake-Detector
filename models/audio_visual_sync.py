import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioVisualSyncModel(nn.Module):
    def _init_(self, audio_features: int = 13, visual_features: int = 512, 
                 hidden_size: int = 256):
        super(AudioVisualSyncModel, self)._init_()
        
        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Visual encoder (for lip region features)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_size),
            nn.ReLU()
        )
        
        # Synchronization classifier
        self.sync_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Sync vs Not Sync
        )
        
    def forward(self, audio_features, lip_images):
        # Encode audio and visual features
        audio_encoded = self.audio_encoder(audio_features)
        visual_encoded = self.visual_encoder(lip_images)
        
        # Concatenate features
        combined = torch.cat([audio_encoded, visual_encoded], dim=1)
        
        # Classify synchronization
        sync_output = self.sync_classifier(combined)
        
        return sync_output