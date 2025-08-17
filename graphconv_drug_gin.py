import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, LeakyReLU, Tanh, Embedding
from torch_geometric.nn import global_add_pool, BatchNorm, EdgeConv, TopKPooling
from torch_geometric.nn import GINConv
import math
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import DynamicEdgeConv
from DGCNN import EdgeConv  # Custom EdgeConv implementation
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
        std = x.std(-1, keepdim=True)     # Standard deviation along feature dimension
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class EdgeDropout(nn.Module):
    """Edge dropout module for graph edges regularization"""
    def __init__(self, keep_prob=0.5):
        super(EdgeDropout, self).__init__()
        assert keep_prob > 0, "Keep probability must be positive"
        self.keep_prob = keep_prob
        self.register_buffer("p", torch.tensor(keep_prob))  # Store as buffer

    def forward(self, edge_index):
        if self.training:
            # Generate random mask based on keep probability
            mask = torch.rand(edge_index.shape[1], device='cuda')
            # Threshold mask to binary values
            mask = torch.floor(mask + self.p).type(torch.bool)
            # Apply mask to edge_index
            edge_index = edge_index[:, mask]
        return edge_index

    def __repr__(self):
        return '{}(keep_prob={})'.format(self.__class__.__name__, self.keep_prob)

class GINConvNet(torch.nn.Module):
    """Graph Neural Network model for molecular property prediction"""
    def __init__(self, n_output=1, num_features_xd=40, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GINConvNet, self).__init__()

        # Dimension configurations
        dim = 256
        hid_dim_gin = 64
        out_dim = 256
        IN_dim = 256
        e_dim = 32
        
        # Edge dropout for regularization
        self.edge_drop = EdgeDropout(keep_prob=1-0.1)
        
        # Embedding layers for atom features
        self.emb1 = Embedding(15+1, e_dim)  # Atom type embedding
        self.emb2 = Embedding(5 + 1, e_dim)  # Feature 1 embedding
        self.emb3 = Embedding(5 + 1, e_dim)  # Feature 2 embedding
        self.emb4 = Embedding(5 + 1, e_dim)  # Feature 3 embedding
        self.emb5 = Embedding(2 + 1, e_dim)  # Feature 4 embedding
        self.edge_emb = Embedding(13 + 1, e_dim)  # Edge type embedding
        
        # Regularization layers
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Model configuration parameters
        self.n_output = n_output
        self.embed_dim = 7
        
        # GIN convolution layers with batch normalization
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.bn_gin1 = BatchNorm1d(hid_dim_gin)
        self.bn_gin2 = BatchNorm1d(hid_dim_gin)
        self.bn_gin3 = BatchNorm1d(hid_dim_gin)
        self.bn_gin4 = BatchNorm1d(hid_dim_gin)
        self.bn_gin5 = BatchNorm1d(out_dim)
        
        # GINConv layers with MLP networks
        nn1 = Sequential(Linear(e_dim*2+1, hid_dim_gin), self.act1, Linear(hid_dim_gin, hid_dim_gin))
        self.conv1 = GINConv(nn1)
        nn2 = Sequential(Linear(hid_dim_gin, out_dim), self.act1, Linear(out_dim, hid_dim_gin))
        self.conv2 = GINConv(nn2)
        nn3 = Sequential(Linear(hid_dim_gin, out_dim), self.act1, Linear(out_dim, hid_dim_gin))
        self.conv3 = GINConv(nn3)
        nn4 = Sequential(Linear(hid_dim_gin, out_dim), self.act1, Linear(out_dim, hid_dim_gin))
        self.conv4 = GINConv(nn4)
        nn5 = Sequential(Linear(out_dim, out_dim), self.act1, Linear(out_dim, hid_dim_gin))
        self.conv5 = GINConv(nn5)
        
        # Batch normalization layers
        self.bn6 = BatchNorm1d(output_dim)
        self.bn8 = BatchNorm1d(output_dim)
        self.bn10 = torch.nn.BatchNorm1d(output_dim)
        self.bn128 = torch.nn.BatchNorm1d(128)
        
        # Fully connected layers
        self.last2 = Sequential(Linear(2 * dim, output_dim), ReLU())
        self.fc2_xd = Linear(dim*2, dim)
        self.fc3_xd = Linear(output_dim, output_dim)
        self.fc4_xd = Linear(output_dim*2, output_dim)
        
        # Protein sequence embedding
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.final = nn.Linear(output_dim, output_dim)
        
        # Point cloud processing (EdgeConv) layers
        point_dim = 128
        egconv1 = Sequential(Linear(2 * 3, point_dim), ReLU(), Linear(point_dim, point_dim), ReLU(), BatchNorm1d(point_dim))
        egconv2 = Sequential(Linear(2*point_dim, point_dim), ReLU(), Linear(point_dim, point_dim), ReLU(), BatchNorm1d(point_dim))
        egconv3 = Sequential(Linear(2*point_dim, point_dim), ReLU(), Linear(point_dim, point_dim), ReLU(), BatchNorm1d(point_dim))
        egconv4 = Sequential(Linear(2 * point_dim, point_dim), ReLU(), Linear(point_dim, point_dim), ReLU(), BatchNorm1d(point_dim))
        self.EdgeConv1 = EdgeConv(egconv1, aggr='max')
        self.EdgeConv2 = EdgeConv(egconv2, aggr='max')
        self.EdgeConv3 = EdgeConv(egconv3, aggr='max')
        self.EdgeConv4 = EdgeConv(egconv4, aggr='max')
        
        # Fusion layer for combining features
        self.last3 = Sequential(Linear(dim+point_dim, output_dim), ReLU())
        
        # Final prediction layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)  # Output layer (1 neuron for regression)
        
        # Additional utility layers
        self.func = nn.Linear(127, 32)
        self.dropfinal = nn.Dropout(0.3)
        self.bn1 = BatchNorm1d(64)
        self.bn2 = BatchNorm1d(128)
        self.bn3 = BatchNorm1d(point_dim)
        self.line1 = nn.Linear(self.embed_dim, 32)
        self.line2 = nn.Linear(64, output_dim)
        self.line3 = nn.Linear(32, 32)
        self.line4 = nn.Linear(32, 32)
        self.drop = nn.Dropout(0.2)
        
        # TopK pooling layers for graph downsampling
        self.top0 = TopKPooling(256, ratio=0.9)
        self.top1 = TopKPooling(256, ratio=0.9)
        self.top2 = TopKPooling(256, ratio=0.9)
        self.top3 = TopKPooling(256, ratio=0.8)

    def forward(self, data, data1, func):
        """Forward pass through the network"""
        # Process molecular graph data
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Process 3D structure data (point cloud)
        pos, edge_index_e, type_e, batch_e = data1.x, data1.edge_index, data1.edge_attr, data1.batch
        
        # Apply edge dropout during training
        edge_index = self.edge_drop(edge_index)
        
        # Process atom features
        x = x[:, :-3]  # Remove last 3 features
        x_f1 = self.emb1(torch.squeeze(x[:, 0]))  # Atom type embedding
        x_f2 = self.emb2(x[:, 1]) + self.emb3(x[:, 2]) + self.emb4(x[:, 3])  # Combined embeddings
        x_f3 = x[:, 4] - 1  # Continuous feature adjustment
        x = torch.cat((x_f1, x_f2, x_f3.unsqueeze(1)), 1)  # Concatenate features
        
        # GIN convolution layers
        x1 = self.bn_gin1(F.relu(self.conv1(x, edge_index)))
        x2 = self.bn_gin2(F.relu(self.conv2(x1, edge_index)))
        x3 = self.bn_gin3(F.relu(self.conv3(x2, edge_index)))
        
        # Global max pooling over nodes in each graph
        max1 = gmp(x1, batch)
        max2 = gmp(x2, batch)
        max3 = gmp(x3, batch)
        
        # Transform pooled features
        max5 = self.line2(max3)
        
        # Return intermediate features (actual implementation would continue processing)
        return max5