import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, LeakyReLU, Tanh, Embedding
from torch_geometric.nn import GINConv, global_add_pool, BatchNorm, EdgeConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import DynamicEdgeConv
from DGCNN import EdgeConv  # Custom EdgeConv implementation
from gin import GINEConv  # Custom GINEConv implementation
from torch_geometric.nn import GCNConv, radius_graph
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_sparse import spspmm, coalesce, fill_diag
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)

class my_Attention(nn.Module):
    """Attention mechanism for feature weighting"""
    def __init__(self, in_size=256, hidden_size=64):
        super(my_Attention, self).__init__()
        # Adaptive pooling to reduce spatial dimensions
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # Projection network for attention weights
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False)  # Output single attention weight per feature
        )

    def forward(self, z):
        # Compute attention weights
        w = self.project(z)
        # Normalize weights with softmax
        beta = torch.softmax(w, dim=1)
        # Return weighted sum of features and attention weights
        return (beta * z).sum(1), beta

class SEAttention(nn.Module):
    """Squeeze-and-Excitation attention module for channel-wise feature recalibration"""
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation network
        self.fc = nn.Sequential(
            nn.Linear(channel, reduction),
            nn.Tanh(),
            nn.Linear(reduction, channel, bias=False),
            nn.Sigmoid()  # Sigmoid for [0,1] gating
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global average pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: Compute channel weights
        y = self.fc(y).view(b, c, 1, 1)
        # Scale input features by channel weights
        return x * y.expand_as(x)

class LayerNorm(nn.Module):
    """Custom layer normalization module similar to original Transformer implementation"""
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
        # Normalize and scale
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class GINConvNet(torch.nn.Module):
    """Graph Neural Network model for molecular property prediction using:
    - GINEConv for 2D molecular graphs
    - EdgeConv for 3D point clouds
    - Attention mechanisms for feature fusion"""
    
    def __init__(self, n_output=1, num_features_xd=40, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GINConvNet, self).__init__()

        # Dimension configurations
        dim = 256 - 48
        hid_dim_gin = 256 - dim - 16
        out_dim = 256 - dim
        point_dim = dim
        e_dim = 32
        edge_dim = 15
        
        # Atom feature embeddings
        self.emb1 = Embedding(15 + 1, e_dim)  # Atom type embedding
        self.emb2 = Embedding(5 + 1, e_dim)    # Feature 1 embedding
        self.emb3 = Embedding(5 + 1, e_dim)    # Feature 2 embedding
        self.emb4 = Embedding(5 + 1, e_dim)    # Feature 3 embedding
        self.edge = Embedding(6, edge_dim)      # Edge type embedding
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Model configuration
        self.n_output = n_output
        self.embed_dim = 7  # Dimension for edge features
        
        # GINEConv layers (Graph Isomorphism Network with Edge features)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.bn_gin1 = BatchNorm1d(hid_dim_gin)
        self.bn_gin2 = BatchNorm1d(hid_dim_gin)
        self.bn_gin3 = BatchNorm1d(hid_dim_gin)
        self.bn_gin4 = BatchNorm1d(out_dim)
        self.bn_gin5 = BatchNorm1d(out_dim)
        
        # GINEConv block 1
        nn1 = Sequential(
            Linear(e_dim * 2 + 17, e_dim * 2 + 17), 
            self.act1, 
            Linear(e_dim * 2 + 17, hid_dim_gin)
        )
        self.conv1 = GINEConv(nn1, edge_dim=self.embed_dim, train_eps=True)
        
        # GINEConv block 2
        nn2 = Sequential(
            Linear(out_dim, hid_dim_gin * 4), 
            self.act1, 
            Linear(hid_dim_gin * 4, hid_dim_gin)
        )
        self.conv2 = GINEConv(nn2, edge_dim=self.embed_dim, train_eps=True)
        
        # GINEConv block 3
        nn3 = Sequential(
            Linear(out_dim, hid_dim_gin * 4), 
            self.act1, 
            Linear(hid_dim_gin * 4, hid_dim_gin)
        )
        self.conv3 = GINEConv(nn3, edge_dim=self.embed_dim, train_eps=True)
        
        # GINEConv block 4
        nn4 = Sequential(
            Linear(out_dim, hid_dim_gin), 
            self.act1, 
            Linear(hid_dim_gin, out_dim)
        )
        self.conv4 = GINEConv(nn4, edge_dim=self.embed_dim, train_eps=True)
        
        # Batch normalization layers
        self.bn6 = BatchNorm1d(output_dim)
        self.bn8 = BatchNorm1d(output_dim)
        self.bn10 = nn.BatchNorm1d(256)
        self.bn128 = nn.BatchNorm1d(128)
        
        # Feature fusion layer
        self.last2 = Sequential(Linear(2 * dim, output_dim), ReLU())
        self.final = nn.Linear(output_dim, output_dim)
        
        # EdgeConv layers for 3D point cloud processing
        egconv1 = Sequential(
            Linear(2 * 3, 64),  # Input: coordinate differences
            ReLU(),
            Linear(64, 64)
        )
        self.EdgeConv1 = EdgeConv(egconv1, aggr='max')
        
        egconv2 = Sequential(
            Linear(2 * 64, 256),  # Input: concatenated features
            ReLU(),
            Linear(256, 256)
        )
        self.EdgeConv2 = EdgeConv(egconv2, aggr='max')
        
        egconv3 = Sequential(
            Linear(2 * 256, 256), 
            ReLU(),
            Linear(256, 256)
        )
        self.EdgeConv3 = EdgeConv(egconv3, aggr='max')
        
        egconv4 = Sequential(
            Linear(2 * 256, point_dim), 
            ReLU(),
            Linear(point_dim, point_dim)
        )
        self.EdgeConv4 = EdgeConv(egconv4, aggr='max')
        
        # Feature fusion layer
        self.last3 = Sequential(Linear(dim + point_dim, output_dim), ReLU())
        
        # Batch normalization for EdgeConv outputs
        self.bn1 = BatchNorm1d(64)
        self.bn2 = BatchNorm1d(256)
        self.bn3 = BatchNorm1d(256)
        self.bn4 = BatchNorm1d(point_dim)
        
        # Additional utility layers
        self.line1 = nn.Linear(self.embed_dim, 32)
        self.line2 = nn.Linear(32, 32)
        self.line3 = nn.Linear(32, 32)
        self.line4 = nn.Linear(32, 32)
        self.drop = nn.Dropout(0.1)
        
        # Attention mechanisms
        self.att = nn.Parameter(torch.ones(2, 1, 1) / 2)  # Learnable attention weights
        self.se = my_Attention()  # Custom attention module

    def forward(self, data, data1, func):
        """Forward pass through the network:
        - Processes 2D molecular graph data
        - Processes 3D point cloud data
        - Fuses features from both modalities
        - Returns graph-level embeddings"""
        
        # Unpack 2D molecular graph data
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Unpack 3D point cloud data
        pos, edge_index_e, type_e, batch_e = data1.x, data1.edge_index, data1.edge_attr, data1.batch
        
        # Process 2D molecular features ----------------------------------------
        x = x[:, :-3]  # Remove last 3 features
        
        # Atom type embedding
        x_f1 = self.emb1(torch.squeeze(x[:, 0]))
        # Combined embeddings for atomic properties
        x_f2 = self.emb2(x[:, 1]) + self.emb3(x[:, 2]) + self.emb4(x[:, 3])
        # Continuous feature adjustment
        x_f3 = x[:, 4] - 1
        
        # Process edge features
        edge_attr = edge_type[:, :self.embed_dim]  # First part of edge attributes
        edge_type1 = self.edge(edge_attr[:, 0])    # Embed bond type
        # Combine bond type embedding with bond distance
        edge_attr = torch.cat((edge_type1, (edge_attr[:, 1] - 1).unsqueeze(1)), 1)
        
        # Concatenate atom features
        x = torch.cat((x_f1, x_f2, x_f3.unsqueeze(1)), 1)
        
        # GINEConv layers with edge features
        x1 = self.bn_gin1(F.relu(self.conv1(x, edge_index, edge_attr)))
        x2 = self.bn_gin2(F.relu(self.conv2(x1, edge_index, edge_attr)))
        x3 = self.bn_gin3(F.relu(self.conv3(x2, edge_index, edge_attr)))
        x4 = self.bn_gin4(F.relu(self.conv4(x3, edge_index, edge_attr)))
        
        # Process 3D point cloud features --------------------------------------
        # EdgeConv layers
        x_e1 = self.bn1(F.relu(self.EdgeConv1(pos, edge_index_e)))
        x_e2 = self.bn2(F.relu(self.EdgeConv2(x_e1, edge_index_e)))
        x_e3 = self.bn3(F.relu(self.EdgeConv3(x_e2, edge_index_e)))
        x_e4 = self.bn4(F.relu(self.EdgeConv4(x_e3, edge_index_e)))
        
        # Feature fusion ------------------------------------------------------
        # Concatenate 2D graph features and 3D point cloud features
        x_e = torch.cat((x_e4, x4), 1)
        
        # Global pooling: Sum features across all nodes in each graph
        max1 = global_add_pool(x_e, batch)
        max1 =F.relu(max1)
        # Batch normalization of pooled features
        max1 = self.bn6(max1)
        
        return max1
