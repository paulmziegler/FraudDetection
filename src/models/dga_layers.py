import torch
import torch.nn as nn
import dgl
import dgl.function as fn

class DGAGNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_groups=2):
        super(DGAGNNLayer, self).__init__()
        self.num_groups = num_groups
        
        # A separate linear transformation for the node's own features
        self.self_transform = nn.Linear(in_feats, out_feats, bias=False)
        
        # A separate linear transformation for each group of neighbors
        self.neighbor_transforms = nn.ModuleList([
            nn.Linear(in_feats, out_feats, bias=False) for _ in range(num_groups)
        ])

    def forward(self, g, h, group_labels):
        """
        Forward pass with dynamic grouping aggregation.
        
        Args:
            g (dgl.DGLGraph): The graph.
            h (torch.Tensor): Node features.
            group_labels (torch.Tensor): Labels indicating predicted group (0: Benign, 1: Fraud, etc.).
        """
        with g.local_scope():
            g.ndata['h'] = h
            g.ndata['group'] = group_labels
            
            # A custom message function is needed to pass both features and group labels
            g.update_all(self.message_func, self.reduce_func)
            
            # Combine the node's own transformed features with the aggregated neighbor features
            h_self = self.self_transform(h)
            h_neighbors = g.ndata['h_agg']
            
            return h_self + h_neighbors

    def message_func(self, edges):
        """Custom message function to send features and group labels."""
        return {'m': edges.src['h'], 'group': edges.src['group']}

    def reduce_func(self, nodes):
        """
        Custom reduce function to aggregate messages based on their group.
        """
        # nodes.mailbox['m'] has shape (num_nodes, num_neighbors, in_feats)
        # nodes.mailbox['group'] has shape (num_nodes, num_neighbors)
        
        messages = nodes.mailbox['m']
        groups = nodes.mailbox['group']
        
        # Initialize an aggregated feature tensor
        aggregated_feats = torch.zeros(messages.shape[0], self.neighbor_transforms[0].out_features, device=messages.device)
        
        # Apply the correct linear transformation based on the group
        for i in range(self.num_groups):
            # Create a mask for neighbors belonging to the current group
            mask = (groups == i)
            
            if mask.any():
                # Select messages from this group
                group_messages = messages[mask]
                
                # Apply the corresponding transformation
                transformed_messages = self.neighbor_transforms[i](group_messages)
                
                # To sum the results back to the correct nodes, we need a scatter-add operation.
                # DGL's reduce function structure makes this tricky without loops.
                # A simpler, more DGL-idiomatic approach is to pre-transform on edges.
                # However, for clarity and directness, we'll simulate it here.
                # NOTE: This approach is not optimized for performance in a real scenario.
                
                # Find the destination node index for each message
                dst_nodes_for_messages = torch.where(mask)[0]
                
                # Use scatter_add_ to sum messages for each destination node
                aggregated_feats.scatter_add_(0, dst_nodes_for_messages.unsqueeze(1).expand_as(transformed_messages), transformed_messages)

        return {'h_agg': aggregated_feats}
