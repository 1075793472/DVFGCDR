import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, LeakyReLU, Tanh
from torch_geometric.nn import GINConv, global_add_pool, BatchNorm, EdgeConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv, GATConv, DynamicEdgeConv
from DGCNN import EdgeConv  # Custom EdgeConv implementation
from gin import GINEConv  # Custom GINEConv implementation
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)

class LayerNorm(nn.Module):
    """Layer normalization module similar to original Transformer implementation"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # Learnable scaling parameters
        self.a_2 = nn.Parameter(torch.ones(features))
        # Learnable bias parameters
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps  # Small epsilon for numerical stability

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # Mean along feature dimension
        std = x.std(-1, keepdim=True)    # Standard deviation along feature dimension
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class GINConvNet(torch.nn.Module):
    """Graph Neural Network model for molecular property prediction using GINEConv and EdgeConv"""
    def __init__(self, n_output=1, num_features_xd=49, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GINConvNet, self).__init__()

        # Dimension configurations
        dim = 64
        hid_dim = 64
        hid_dim_gin = 64
        
        # Regularization and activation functions
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Model configuration parameters
        self.n_output = n_output
        self.embed_dim = 7  # Dimension for edge features
        
        # GINEConv layers (Graph Isomorphism Network with Edge features)
        self.act1 = nn.LeakyReLU()
        self.act2 = nn.ReLU()
        
        # First GINEConv block
        nn1 = Sequential(
            Linear(num_features_xd, hid_dim_gin),
            self.act1,
            Linear(hid_dim_gin, hid_dim_gin),
            self.act2,
            BatchNorm1d(hid_dim_gin)
        )
        self.conv1 = GINEConv(nn1, edge_dim=self.embed_dim, train_eps=True)
        
        # Second GINEConv block
        nn2 = Sequential(
            Linear(hid_dim_gin, hid_dim_gin),
            self.act1,
            Linear(hid_dim_gin, hid_dim_gin),
            self.act2,
            BatchNorm1d(hid_dim_gin)
        )
        self.conv2 = GINEConv(nn2, edge_dim=self.embed_dim, train_eps=True)
        
        # Third GINEConv block
        nn3 = Sequential(
            Linear(hid_dim_gin, hid_dim_gin),
            self.act1,
            Linear(hid_dim_gin, hid_dim_gin),
            self.act2,
            BatchNorm1d(hid_dim_gin)
        )
        self.conv3 = GINEConv(nn3, edge_dim=self.embed_dim, train_eps=True)
        
        # Fourth GINEConv block
        nn4 = Sequential(
            Linear(hid_dim_gin, hid_dim_gin),
            self.act1,
            Linear(hid_dim_gin, hid_dim_gin),
            self.act2,
            BatchNorm1d(hid_dim_gin)
        )
        self.conv4 = GINEConv(nn4, edge_dim=self.embed_dim, train_eps=True)
        
        # Batch normalization layers
        self.bn8 = BatchNorm1d(output_dim)
        self.bn10 = nn.BatchNorm1d(128)
        self.bn128 = nn.BatchNorm1d(128)
        
        # Protein sequence embedding layer
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        
        # Point cloud processing (EdgeConv) layers
        point_dim = 256
        self.final = nn.Linear(point_dim, output_dim)
        self.final2 = Sequential(
            Linear(point_dim, point_dim), 
            ReLU(),  
            BatchNorm1d(point_dim)
        )
        
        # EdgeConv blocks for processing 3D point cloud data
        egconv1 = Sequential(
            Linear(2 * 3, 128),  # Input features: concatenated coordinate differences
            ReLU(),
            Linear(128, 128)
        )
        self.EdgeConv1 = EdgeConv(egconv1, aggr='max')
        
        egconv2 = Sequential(
            Linear(2 * 128, 256),  # Input features: concatenated previous layer outputs
            ReLU(),
            Linear(256, 256)
        )
        self.EdgeConv2 = EdgeConv(egconv2, aggr='max')
        
        egconv3 = Sequential(
            Linear(2 * point_dim, point_dim),
            ReLU(),
            Linear(point_dim, point_dim)
        )
        self.EdgeConv3 = EdgeConv(egconv3, aggr='max')
        
        egconv4 = Sequential(
            Linear(2 * point_dim, point_dim),
            ReLU(),
            Linear(point_dim, point_dim)
        )
        self.EdgeConv4 = EdgeConv(egconv4, aggr='max')
        
        # Additional layers
        self.last3 = Sequential(
            Linear(point_dim, output_dim), 
            ReLU(),
            BatchNorm1d(output_dim)
        )
        self.func = nn.Linear(127, 32)  # Functional group feature processing
        self.dropfinal = nn.Dropout(0.3)  # Final dropout layer
        
        # Batch normalization layers for EdgeConv outputs
        self.bn1 = BatchNorm1d(128)
        self.bn2 = BatchNorm1d(256)
        self.bn3 = BatchNorm1d(point_dim)
        self.bn4 = BatchNorm1d(point_dim)

    def forward(self, data, data1, func):
        """Forward pass through the network"""
        # Process molecular graph data (2D structure)
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Process 3D structure data (point cloud)
        pos, edge_index_e, type_e, batch_e = data1.x, data1.edge_index, data1.edge_attr, data1.batch
        
        # Remove last 3 features from 2D molecular data
        x = x[:, :-3]
        
        # Process 3D point cloud data with EdgeConv layers
        # Layer 1: Process raw coordinates
        x_e1 = self.bn1(F.relu(self.EdgeConv1(pos, edge_index)))
        # Layer 2: Process features from first layer
        x_e2 = self.bn2(F.relu(self.EdgeConv2(x_e1, edge_index)))
        # Layer 3: Process features from second layer
        x_e3 = self.bn3(F.relu(self.EdgeConv3(x_e2, edge_index)))
        # Layer 4: Process features from third layer
        x_e4 = self.bn4(F.relu(self.EdgeConv4(x_e3, edge_index)))
        
        # Global pooling: Sum features across all nodes in each graph
        max2 = global_add_pool(x_e4, batch)
        
        # Apply batch normalization to the pooled features
        x = self.bn8(max2)
        
        return x
