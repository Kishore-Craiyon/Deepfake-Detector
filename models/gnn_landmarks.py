import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

class LandmarkGNN(nn.Module):
    def _init_(self, node_features: int = 2, hidden_channels: int = 128, 
                 num_layers: int = 3, num_classes: int = 2):
        super(LandmarkGNN, self)._init_()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(node_features, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        # Last layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def create_face_graph(self, landmarks):
        """Create graph structure for facial landmarks"""
        # Define edges based on facial structure
        # This is a simplified version - you can make it more sophisticated
        num_points = landmarks.shape[0]
        
        # Connect adjacent points in facial features
        edges = []
        
        # Jaw line (0-16)
        for i in range(16):
            edges.append([i, i+1])
            
        # Right eyebrow (17-21)
        for i in range(17, 21):
            edges.append([i, i+1])
            
        # Left eyebrow (22-26)
        for i in range(22, 26):
            edges.append([i, i+1])
            
        # Nose (27-35)
        for i in range(27, 35):
            edges.append([i, i+1])
            
        # Right eye (36-41)
        for i in range(36, 41):
            edges.append([i, i+1])
        edges.append([41, 36])  # Close the eye
        
        # Left eye (42-47)
        for i in range(42, 47):
            edges.append([i, i+1])
        edges.append([47, 42])  # Close the eye
        
        # Lips (48-67)
        for i in range(48, 67):
            edges.append([i, i+1])
        edges.append([67, 48])  # Close the lip
        
        edge_index = torch.tensor(edges).t().contiguous()
        return edge_index
        
    def forward(self, landmarks_batch):
        # landmarks_batch: (batch_size, sequence_length, 68, 2)
        batch_size, seq_len, num_landmarks, coords = landmarks_batch.shape
        
        graphs = []
        for b in range(batch_size):
            for t in range(seq_len):
                landmarks = landmarks_batch[b, t]  # (68, 2)
                edge_index = self.create_face_graph(landmarks)
                
                graph = Data(x=landmarks, edge_index=edge_index)
                graphs.append(graph)
        
        # Batch graphs
        batch = Batch.from_data_list(graphs)
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        
        # Apply GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch_idx)
        
        # Reshape back to (batch_size, seq_len, hidden_channels)
        x = x.view(batch_size, seq_len, -1)
        
        # Use mean across sequence for classification
        x = torch.mean(x, dim=1)
        
        output = self.classifier(x)
        return output