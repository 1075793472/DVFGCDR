# python3
# -*- coding:utf-8 -*-

import torch
from  process import *
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr,spearmanr
import time
import pickle
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from torch.utils.data import Dataset, DataLoader
import torch_geometric as pyg
import joblib
class TF(nn.Module):
    def __init__(self, in_features,hidden_features=512,act_layer=nn.ReLU,drop=0.):
        super().__init__()
        self.Block1 = kaggle(in_features=in_features, hidden_features=hidden_features, act_layer= act_layer, drop=drop, pred=False)
        self.Block2 = kaggle(in_features=in_features, hidden_features=hidden_features, act_layer= act_layer, drop=drop, pred=True)
    def forward(self, x):
            return self.Block2(self.Block1(x))

# Set device configuration (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess gene expression data
rnadata = pd.read_csv('Cell_line_RMA_proc_basalExp.txt', sep='\t')
rnadata = rnadata.drop(['GENE_SYMBOLS', 'GENE_title'], axis=1, inplace=False)  # Remove metadata columns
knn = np.array(rnadata.values.T)  # Transpose to cell lines × genes

# Load pre-trained PCA model for dimensionality reduction
GENE_NUM = 1018  # Number of genes after preprocessing
pca = joblib.load("pca3.m")  # Load pre-trained PCA model
features = pca.transform(knn)  # Apply PCA transformation

# Standardize features (z-score normalization)
features = (features - np.mean(features)) / (np.std(features))
knn = features  # Store processed features

class MolData(Dataset):
    """Custom Dataset for drug sensitivity data
    Integrates drug molecular graphs with cell line gene expression data"""
    def __init__(self, path):
        super(MolData, self).__init__()
        # Load drug molecular graphs
        self.data = torch.load('g1_' + path + '.pt')  # 2D molecular graph
        self.data1 = torch.load('g2_' + path + '.pt')  # 3D molecular graph (optional)
        self.rna = knn  # Gene expression matrix (cell lines × genes)
        print(f'Data loaded for {path}')

    def __getitem__(self, index):
        """Retrieve single data sample"""
        y = self.data[index]['y']           # Target value (drug sensitivity IC50)
        g1 = self.data[index]               # 2D molecular graph data
        g2 = self.data1[index]              # 3D molecular graph data
        rna = self.rna[self.data[index]['target']]  # Gene features for specific cell line
        func = self.data[index]['func']     # Functional features
        return y, g1, g2, rna, func

    def __len__(self):
        """Number of samples in dataset"""
        return len(self.data)

def my_collate(samples):
    """Custom collate function for batching graph data
    Handles variable-sized graph structures and missing 3D graphs"""
    # Unpack samples
    y = [s[0] for s in samples]  # Target values
    g1 = [s[1] for s in samples]  # 2D graphs
    g2 = [s[2] for s in samples]  # 3D graphs
    
    # Convert to tensors
    rna = torch.FloatTensor(np.array([s[3] for s in samples]))  # Gene expression
    func = torch.FloatTensor(np.array([s[4] for s in samples]))  # Functional features
    y = torch.cat(y, dim=0)  # Concatenate target values
    
    # Batch graph data using PyG's batching utility
    G1 = pyg.data.Batch().from_data_list(g1)
    
    # Handle missing 3D graphs
    if None in g2:  
        return y, G1, None
    else:
        G2 = pyg.data.Batch().from_data_list(g2)
        return y, G1, G2, rna, func

class MLP1(nn.Sequential):
    """Multi-Layer Perceptron for processing gene expression features"""
    def __init__(self):
        super(MLP1, self).__init__()
        # Network architecture
        self.line = nn.Linear(GENE_NUM, GENE_NUM)  # Input layer
        self.line2 = nn.Linear(GENE_NUM, 256)      # Bottleneck layer
        self.drop = nn.Dropout(0.2)                # Regularization
        self.act = nn.ReLU()                       # Activation function
    
    def forward(self, rna):
        """Process gene expression data through MLP"""
        rna = F.relu(self.line(rna))
        rna = F.relu(self.line2(rna))
        return rna

class kaggle(nn.Module):
    """Transformer block with self-attention mechanism
    Inspired by Kaggle competition architectures"""
    def __init__(self, in_features, hidden_features=256, act_layer=nn.GELU, drop=0.1, pred=True):
        super().__init__()
        # Attention mechanism components
        self.q = nn.Linear(in_features, in_features)  # Query projection
        self.k = nn.Linear(in_features, in_features)  # Key projection
        self.v = nn.Linear(in_features, in_features)  # Value projection
        
        # Feed-forward network
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()  # Activation function (GELU by default)
        self.pred = pred  # Flag for prediction mode
        
        # Output projection
        if pred:
            self.fc2 = nn.Linear(hidden_features, 256)  # Prediction head
        else:
            self.fc2 = nn.Linear(hidden_features, in_features)  # Feature transformation
        
        self.drop = nn.Dropout(0.1)  # Dropout for regularization

    def forward(self, x):
        """Transformer forward pass with residual connections"""
        # Self-attention mechanism
        x0 = x  # Residual connection
        q = self.q(x).unsqueeze(2)
        k = self.k(x).unsqueeze(2)
        v = self.v(x).unsqueeze(2)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)  # Softmax normalization
        
        # Context vector
        x = (attn @ v).squeeze(2)
        x += x0  # First residual connection
        
        # Feed-forward network
        x1 = x  # Second residual connection
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        
        # Second residual connection (only in feature extraction mode)
        if not self.pred:
            x += x1
        
        return x.squeeze(0)

# Fixed drug feature dimension
drug_dim = 256

class Classifier(nn.Sequential):
    """Main classification/regression model
    Combins drug and gene features for sensitivity prediction"""
    def __init__(self, model_drug, model_gene):
        super(Classifier, self).__init__()
        # Feature extraction modules
        self.model_drug = model_drug  # Drug feature extractor (GNN)
        self.model_gene = model_gene  # Gene feature extractor (MLP)
        self.dropout = nn.Dropout(0.2)  # Regularization
        
        # Prediction network architecture
        self.hidden_dims = [1024, 1024, 512]  # Hidden layer dimensions
        layer_size = len(self.hidden_dims) + 1
        # Dimension progression: [input, hidden..., output]
        dims = [drug_dim + 256] + self.hidden_dims + [1]  
        
        # Create fully connected layers
        self.predictor = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)]
        )
    
    def forward(self, G1, G2, rna, func):
        """End-to-end forward pass:
        1. Extract drug features from molecular graphs
        2. Extract gene features from expression data
        3. Concatenate features
        4. Predict sensitivity through fully-connected network"""
        # Drug feature extraction
        v_D = self.model_drug(G1, G2, func)
        # Gene feature extraction
        v_P = self.model_gene(rna)
        # Feature fusion
        v_f = torch.cat((v_D, v_P), 1)
        
        # Pass through predictor network
        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):  # Output layer (no activation)
                v_f = l(v_f)
            else:  # Hidden layers (ReLU + dropout)
                v_f = F.relu(self.dropout(l(v_f)))
        return v_f

# Import DVFGCONV extractor (from separate module)
from graphconv_drug import GINConvNet

class DVFGCONV:
    """Main model class for Drug Sensitivity Prediction
    Integrates Graph Convolutional Networks and Gene Expression analysis"""
    def __init__(self, modeldir):
        # Initialize submodels
        model_drug = GINConvNet(output_dim=drug_dim)  # Drug feature extractor (GNN)
        model_gene = MLP1()                           # Gene feature extractor (MLP)
        
        # Combine into classifier
        self.model = Classifier(model_drug, model_gene)  
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.modeldir = modeldir  # Directory for saving models/results
        
        # Output files configuration
        self.record_file = os.path.join(self.modeldir, "valid_markdowntable.txt")  # Validation metrics
        self.pkl_file = os.path.join(self.modeldir, "loss_curve_iter.pkl")          # Loss history
    
    def test(self, datagenerator, model):
        """Model evaluation method
        Computes various regression metrics on test data"""
        y_label = []  # Ground truth labels
        y_pred = []   # Model predictions
        model.eval()  # Set evaluation mode
        
        for batch_idx, (y, g1, g2, rna, func) in enumerate(datagenerator):
            # Forward pass
            score = model(
                g1.to(device), 
                g2.to(device) if g2 else None, 
                rna.to(device), 
                func.to(device)
            )
            
            # Process labels and predictions
            label = y.view(-1).float().to(device)
            
            # Loss calculation (MSE)
            loss_fct = torch.nn.MSELoss()
            n = torch.squeeze(score, 1).float()
            loss = loss_fct(n, label)
            
            # Collect results for metrics
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            y_label.extend(label_ids.flatten().tolist())
            y_pred.extend(logits.flatten().tolist())
        
        model.train()  # Restore training mode
        
        # Calculate regression metrics
        mse = mean_squared_error(y_label, y_pred)
        rmse = np.sqrt(mse)
        pearson_corr, pearson_p = pearsonr(y_label, y_pred)
        spearman_corr, spearman_p = spearmanr(y_label, y_pred)
        ci = concordance_index(y_label, y_pred)  # Concordance Index
        
        return y_label, y_pred, mse, rmse, pearson_corr, pearson_p, \
               spearman_corr, spearman_p, ci, loss
    
    def train(self, train_drug, val_drug):
        """Model training procedure with:
        - Learning rate scheduling
        - Early stopping
        - TensorBoard logging
        - Validation metrics tracking"""
        # Hyperparameters
        lr = 0.0001        # Initial learning rate
        decay = 0.0002      # Weight decay (L2 regularization)
        BATCH_SIZE = 128    # Batch size
        train_epoch = 300   # Total training epochs
        
        # Device setup
        self.model = self.model.to(self.device)
        
        # Optimizer (Adam with weight decay)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
        loss_history = []  # Loss tracking
        
        # Data loaders with custom collation
        training_generator = DataLoader(
            train_drug, 
            collate_fn=my_collate, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            pin_memory=True
        )
        validation_generator = DataLoader(
            val_drug, 
            collate_fn=my_collate, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            pin_memory=True
        )
        
        # Model checkpointing setup
        max_MSE = 10000  # Initialize with high value
        valid_metric_record = []  # Validation metrics storage
        
        # Reporting setup (PrettyTable for console output)
        valid_metric_header = [
            '# epoch', "MSE", 'RMSE', "Pearson Correlation", "with p-value", 
            'Spearman Correlation', "with p-value2", "Concordance Index"
        ]
        table = PrettyTable(valid_metric_header)
        float2str = lambda x: '%0.4f' % x  # Formatting helper
        
        # Training initialization
        print('--- Starting Training ---')
        writer = SummaryWriter(self.modeldir, comment='Drug_Transformer_MLP')  # TensorBoard
        t_start = time.time()
        iteration_loss = 0  # Global iteration counter
        
        # Epoch loop
        for epo in range(train_epoch):
            loss_sum = 0  # Epoch loss accumulator
            

            # Batch iteration
            for batch_idx, (y, g1, g2, rna, func) in enumerate(training_generator):
                # Forward pass
                score = self.model(
                    g1.to(device, non_blocking=True), 
                    g2.to(device, non_blocking=True) if g2 else None, 
                    rna.to(device, non_blocking=True), 
                    func.to(device, non_blocking=True)
                )
                
                # Loss calculation
                label = y.to(device, non_blocking=True)
                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(score, 1).float()
                loss = loss_fct(n, label)
                
                # Record and log loss
                loss_history.append(loss.item())
                writer.add_scalar("Loss/train", loss.item(), iteration_loss)
                iteration_loss += 1
                loss_sum += loss.item()
                
                # Backpropagation
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            
            # Epoch summary
            print(f'Epoch {epo+1}/{train_epoch} | Train Loss: {loss_sum:.4f}')
            
            # Validation phase
            with torch.no_grad():
                # Run evaluation
                y_true, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI, loss_val = \
                    self.test(validation_generator, self.model)
                
                # Print metrics
                print(f'Validation | MSE: {mse:.4f} | Pearson: {person:.4f} | CI: {CI:.4f}')
                
                # Format metrics for table
                lst = ["epoch " + str(epo)] + list(map(float2str, [
                    mse, rmse, person, p_val, spearman, s_p_val, CI
                ]))
                valid_metric_record.append(lst)
                
                # Update best model (early stopping based on MSE)
                if mse < max_MSE:
                    max_MSE = mse
                    print(f'New best model at epoch {epo+1} with MSE: {mse:.4f}')
                    
                    # Log validation metrics to TensorBoard
                    writer.add_scalar("valid/mse", mse, epo)
                    writer.add_scalar('valid/rmse', rmse, epo)
                    writer.add_scalar("valid/pearson", person, epo)
                    writer.add_scalar("valid/concordance_index", CI, epo)
                    writer.add_scalar("valid/spearman", spearman, epo)
                    writer.add_scalar("Loss/valid", loss_val.item(), iteration_loss)
            
            # Add to results table
            table.add_row(lst)
        
        # Save training artifacts
        os.makedirs(self.modeldir, exist_ok=True)
        with open(self.record_file, 'w') as fp:
            fp.write(table.get_string())
        with open(self.pkl_file, 'wb') as pck:
            pickle.dump(loss_history, pck)
        
        # Finalize training
        duration = time.time() - t_start
        print(f'--- Training Completed in {duration:.2f} seconds ---')
        writer.flush()
        writer.close()
    
    def save_model(self):
        """Save model state dictionary to file"""
        save_path = os.path.join(self.modeldir, 'model.pt')
        torch.save(self.model.state_dict(), save_path)
        print(f'Model saved to {save_path}')
    
    def load_pretrained(self, path):
        """Load pre-trained model weights
        Handles GPU/CPU compatibility and DataParallel wrappers"""
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Device-compatible loading
        if self.device.type == 'cuda':
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
        
        # Handle DataParallel naming (remove 'module.' prefix)
        if next(iter(state_dict)).startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # Load state dict
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        print(f'Pre-trained model loaded from {path}')


if __name__ == '__main__':
    #If want to re-splitting data
    from Step2_DataEncoding import DataEncoding
    
    vocab_dir = '/home/s3540/DTI/DVFGCONV-main'
    obj = DataEncoding(vocab_dir=vocab_dir)
    traindata, testdata = obj.Getdata.ByCancer(random_seed=2)
    obj.encode(
     traindata=traindata,
     testdata=testdata)
    
    modeldir = 'Model_80'  # Output directory
    modelfile = os.path.join(modeldir, 'model.pt')  # Model save path
    
    # Load datasets
    traindata = MolData('train')  # Training dataset
    testdata = MolData('test')    # Validation dataset
    
    # Initialize model
    net = DVFGCONV(modeldir=modeldir)
    
    # Train model
    net.train(train_drug=traindata, val_drug=testdata)
    
    # Save final model
    net.save_model()
    print(f"Model saved: {modelfile}")
