import torch.nn as nn
import torch.nn.functional as F
from .dga_layers import DGAGNNLayer

class DGAGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_groups=2):
        super(DGAGNN, self).__init__()
        self.layer1 = DGAGNNLayer(in_feats, h_feats, num_groups)
        self.layer2 = DGAGNNLayer(h_feats, h_feats, num_groups)
        self.classify = nn.Linear(h_feats, num_classes)
        
    def forward(self, g, h, group_labels):
        """
        Forward pass for the full DGA-GNN model.
        
        Args:
            g (dgl.DGLGraph): The graph.
            h (torch.Tensor): Node features.
            group_labels (torch.Tensor): Node group labels from previous epoch/init.
        """
        # Apply first layer and a non-linear activation
        h = F.relu(self.layer1(g, h, group_labels))
        
        # Apply second layer and a non-linear activation
        h = F.relu(self.layer2(g, h, group_labels))
        
        # Apply final classification layer
        return self.classify(h)
