import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

class SpatialCNN(nn.Module):
    def __init__(self, backbone: str = 'xception', num_classes: int = 2, 
                 dropout_rate: float = 0.5, pretrained: bool = True):
        super(SpatialCNN, self).__init__()
        
        self.backbone_name = backbone
        
        if backbone == 'xception':
            # Note: For Xception, you might need to install timm
            # pip install timm
            import timm
            self.backbone = timm.create_model('xception', pretrained=pretrained)
            feature_dim = self.backbone.num_features
            self.backbone.classifier = nn.Identity()
            
        elif backbone == 'inception_v3':
            self.backbone = models.inception_v3(pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        if hasattr(features, 'logits'):  # For inception_v3
            features = features.logits
        output = self.classifier(features)
        return output, features  # Return both output and features