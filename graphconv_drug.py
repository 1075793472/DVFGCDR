import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, LeakyReLU
from torch_geometric.nn import GINConv, global_add_pool, BatchNorm
from torch_geometric.nn.conv import PointConv
from DGCNN import EdgeConv  # Custom EdgeConv implementation
from gin import GINEConv     # Custom GINEConv implementation

# Set random seeds for reproducibility
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)

class LayerNorm(nn.Module):
    """Layer normalization module similar to the original Transformer paper.
    
    Args:
        features (int): Number of features in the input
        eps (float): Small epsilon value for numerical stability
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # Scaling parameter
        self.b_2 = nn.Parameter(torch.zeros(features))  # Shifting parameter
        self.eps = eps  # Small constant for numerical stability

    def forward(self, x):
        """Apply layer normalization to input tensor.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, ..., features]
            
        Returns:
            Tensor: Normalized output tensor
        """
        mean = x.mean(-1, keepdim=True)  # Mean along last dimension
        std = x.std(-1, keepdim=True)    # Standard deviation along last dimension
        # Normalize and apply affine transformation
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class GINConvNet(torch.nn.Module):
    """Graph Isomorphism Network (GIN) with point cloud processing.
    
    This model processes both molecular graphs (using GIN layers) and 
    3D molecular structures (using EdgeConv layers), combining information
    from both representations.
    
    Args:
        n_output (int): Number of output units
        num_features_xd (int): Number of atom features
        num_features_xt (int): Number of protein sequence features
        n_filters (int): Number of filters in convolutional layers
        embed_dim (int): Embedding dimension
        output_dim (int): Output dimension of the model
        dropout (float): Dropout rate
    """
    
    def __init__(self, n_output=1, num_features_xd=49, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GINConvNet, self).__init__()

        # Initialize dimensions and activation functions
        dim = 64
        hid_dim = 64
        hid_dim_gin = 64
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        self.embed_dim = 7  # Edge embedding dimension
        self.act1 = nn.LeakyReLU()  # First activation function
        self.act2 = nn.ReLU()        # Second activation function

        # Graph Isomorphism Network (GIN) layers
        # GIN layer 1
        nn1 = Sequential(
            Linear(num_features_xd, hid_dim_gin), self.act1,
            Linear(hid_dim_gin, hid_dim_gin), self.act2,
            BatchNorm1d(hid_dim_gin)
        )
        self.conv1 = GINEConv(nn1, edge_dim=self.embed_dim, train_eps=True)
        
        # GIN layer 2
        nn2 = Sequential(
            Linear(hid_dim_gin, hid_dim_gin), self.act1,
            Linear(hid_dim_gin, hid_dim_gin), self.act2,
            BatchNorm1d(hid_dim_gin)
        )
        self.conv2 = GINEConv(nn2, edge_dim=self.embed_dim, train_eps=True)
        
        # GIN layer 3
        nn3 = Sequential(
            Linear(hid_dim_gin, hid_dim_gin), self.act1,
            Linear(hid_dim_gin, hid_dim_gin), self.act2,
            BatchNorm1d(hid_dim_gin)
        )
        self.conv3 = GINEConv(nn3, edge_dim=self.embed_dim, train_eps=True)
        
        # GIN layer 4
        nn4 = Sequential(
            Linear(hid_dim_gin, hid_dim_gin), self.act1,
            Linear(hid_dim_gin, hid_dim_gin), self.act2,
            BatchNorm1d(hid_dim_gin)
        )
        self.conv4 = GINEConv(nn4, edge_dim=self.embed_dim, train_eps=True)

        # Batch normalization layers
        self.bn8 = BatchNorm1d(output_dim)
        self.bn10 = BatchNorm1d(128)
        self.bn128 = BatchNorm1d(128)
        
        # Linear layer for combining representations
        self.last2 = Sequential(
            Linear(2 * dim, output_dim), 
            ReLU()
        )

        # Protein sequence embedding layer
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)

        # Point cloud processing parameters
        point_dim = 256
        
        # Final layers for point cloud processing
        self.final = nn.Linear(point_dim, output_dim)
        self.final2 = Sequential(
            Linear(point_dim, point_dim), 
            ReLU(),  
            BatchNorm1d(point_dim)
        )
        
        # Edge convolution modules for point cloud processing
        # EdgeConv layer 1
        egconv1 = Sequential(
            Linear(2 * 3, 128),
            ReLU(),
            Linear(128, 128)
        )
        self.EdgeConv1 = EdgeConv(egconv1, aggr='max')
        
        # EdgeConv layer 2
        egconv2 = Sequential(
            Linear(2 * 128, 256),
            ReLU(),
            Linear(256, 256)
        )
        self.EdgeConv2 = EdgeConv(egconv2, aggr='max')
        
        # EdgeConv layer 3
        egconv3 = Sequential(
            Linear(2 * point_dim, point_dim),
            ReLU(),
            Linear(point_dim, point_dim)
        )
        self.EdgeConv3 = EdgeConv(egconv3, aggr='max')
        
        # EdgeConv layer 4
        egconv4 = Sequential(
            Linear(2 * point_dim, point_dim),
            ReLU(),
            Linear(point_dim, point_dim)
        )
        self.EdgeConv4 = EdgeConv(egconv4, aggr='max')
        
        # Final processing layers
        self.last3 = Sequential(
            Linear(point_dim, output_dim), 
            ReLU(),
            BatchNorm1d(output_dim)
        )
        
        # Functional group processing
        self.func = nn.Linear(127, 32)
        self.dropfinal = nn.Dropout(0.3)
        
        # Additional batch normalization layers
        self.bn1 = BatchNorm1d(128)
        self.bn2 = BatchNorm1d(256)
        self.bn3 = BatchNorm1d(point_dim)
        self.bn4 = BatchNorm1d(point_dim)

    def forward(self, data, data1, func):
        """
        Forward pass through the network.
        
        Args:
            data: PyG Data object for molecular graph
            data1: PyG Data object for 3D point cloud
            func: Functional group features
            
        Returns:
            Tensor: Output representation of the molecule
        """
        # Unpack graph data
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Unpack point cloud data
        pos, edge_index_e, type_e, batch_e = data1.x, data1.edge_index, data1.edge_attr, data1.batch
        
        # Process only the atom features (exclude coordinates)
        x = x[:, :-3]
        
        # Process point cloud with EdgeConv layers
        # First EdgeConv layer
        x_e1 = self.bn1(F.relu(self.EdgeConv1(pos, edge_index)))
        
        # Second EdgeConv layer
        x_e2 = self.bn2(F.relu(self.EdgeConv2(x_e1, edge_index)))
        
        # Third EdgeConv layer
        x_e3 = self.bn3(F.relu(self.EdgeConv3(x_e2, edge_index)))
        
        # Fourth EdgeConv layer
        x_e4 = self.bn4(F.relu(self.EdgeConv4(x_e3, edge_index)))
        
        # Global pooling (sum over all nodes in each graph)
        max2 = global_add_pool(x_e4, batch)
        
        # Final normalization
        x = self.bn8(max2)
        
        return x