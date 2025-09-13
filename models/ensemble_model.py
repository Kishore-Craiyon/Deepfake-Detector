import torch
import torch.nn as nn
from .spatial_cnn import SpatialCNN
from .temporal_lstm import TemporalLSTM
from .gnn_landmarks import LandmarkGNN
from .audio_visual_sync import AudioVisualSyncModel

class DeepFakeEnsemble(nn.Module):
    def _init_(self, config):
        super(DeepFakeEnsemble, self)._init_()
        
        # Initialize sub-models
        self.spatial_cnn = SpatialCNN(
            backbone=config['model']['spatial']['backbone'],
            num_classes=2,
            dropout_rate=config['model']['spatial']['dropout_rate']
        )
        
        self.temporal_lstm = TemporalLSTM(
            input_size=512,  # Features from spatial CNN
            hidden_size=config['model']['temporal']['hidden_size'],
            num_layers=config['model']['temporal']['num_layers'],
            dropout_rate=config['model']['temporal']['dropout_rate']
        )
        
        self.landmark_gnn = LandmarkGNN(
            node_features=2,
            hidden_channels=config['model']['gnn']['hidden_channels'],
            num_layers=config['model']['gnn']['num_layers']
        )
        
        self.sync_model = AudioVisualSyncModel(
            audio_features=config['model']['audio_visual']['audio_features'],
            visual_features=config['model']['audio_visual']['visual_features']
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(8, 64),  # 4 models * 2 classes each
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, batch_data):
        """
        batch_data should contain:
        - 'frames': video frames for spatial analysis
        - 'sequences': frame sequences for temporal analysis  
        - 'landmarks': facial landmarks for GNN
        - 'audio': audio features
        - 'lip_regions': lip region images
        """
        
        # Spatial analysis
        spatial_out, spatial_features = self.spatial_cnn(batch_data['frames'])
        
        # Temporal analysis
        temporal_out, _ = self.temporal_lstm(batch_data['sequences'])
        
        # Landmark analysis
        landmark_out = self.landmark_gnn(batch_data['landmarks'])
        
        # Audio-visual sync
        sync_out = self.sync_model(batch_data['audio'], batch_data['lip_regions'])
        
        # Combine all outputs
        all_outputs = torch.cat([
            torch.softmax(spatial_out, dim=1),
            torch.softmax(temporal_out, dim=1),
            torch.softmax(landmark_out, dim=1),
            torch.softmax(sync_out, dim=1)
        ], dim=1)
        
        # Final fusion
        final_output = self.fusion(all_outputs)
        
        return {
            'final_output': final_output,
            'spatial_output': spatial_out,
            'temporal_output': temporal_out,
            'landmark_output': landmark_out,
            'sync_output': sync_out
        }