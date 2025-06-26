# python3
# -*- coding:utf-8 -*-

import torch
from  process import *
import os
import numpy as np
import pandas as pd
from torch_geometric.nn import GATConv
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
class TF(nn.Module):
    def __init__(self, in_features,hidden_features=512,act_layer=nn.ReLU,drop=0.):
        super().__init__()
        self.Block1 = kaggle(in_features=in_features, hidden_features=hidden_features, act_layer= act_layer, drop=drop, pred=False)
        self.Block2 = kaggle(in_features=in_features, hidden_features=hidden_features, act_layer= act_layer, drop=drop, pred=True)
    def forward(self, x):
            return self.Block2(self.Block1(x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rnadata = pd.read_csv('Cell_line_RMA_proc_basalExp.txt', sep='\t')
rnadata = rnadata.drop(['GENE_SYMBOLS', 'GENE_title'], axis=1, inplace=False)
knn = np.array(rnadata.values.T)
K =10
import joblib
GENE_NUM=1018
pca=joblib.load("pca3.m")
features = pca.transform(knn)
#
features = (features - np.mean(features)) / (np.std(features))
knn=features
# Dataset class for drug sensitivity data
class MolData(Dataset):
    def __init__(self, path):
        super(MolData, self).__init__()
        # Load drug graph data
        self.data = torch.load('g1_' + path + '.pt')  # Primary drug 2d graph
        self.data1 = torch.load('g2_' + path + '.pt')  # Secondary drug 3d graph
        self.rna = knn  # Gene expression matrix
        print(f'Data loaded for {path}')

    def __getitem__(self, index):
        # Retrieve data elements
        y = self.data[index]['y']          # Target value (drug sensitivity)
        g1 = self.data[index]              # Primary drug 2d graph
        g2 = self.data1[index]             # Secondary drug 3d graph
        # Cell line gene features
        rna = self.rna[self.data[index]['target']]  
        func = self.data[index]['func']    # Functional features
        return y, g1, g2, rna, func

    def __len__(self):
        return len(self.data)

# Custom collate function for batching graph data
def my_collate(samples):
    # Unpack samples
    y = [s[0] for s in samples]  
    g1 = [s[1] for s in samples]  
    g2 = [s[2] for s in samples]  
    # Convert to tensors
    rna = torch.FloatTensor(np.array([s[3] for s in samples]))  
    func = torch.FloatTensor(np.array([s[4] for s in samples]))  
    y = torch.cat(y, dim=0)  # Concatenate target values
    
    # Batch graph data
    G1 = pyg.data.Batch().from_data_list(g1)
    # Handle missing secondary graphs
    if None in g2:  
        return y, G1, None
    else:
        G2 = pyg.data.Batch().from_data_list(g2)
        return y, G1, G2, rna, func

# Gene Feature Processing Network
class MLP1(nn.Sequential):
    def __init__(self):
        super(MLP1, self).__init__()
        # Define network architecture
        self.line = nn.Linear(GENE_NUM, GENE_NUM)  # Input layer
        self.line2 = nn.Linear(GENE_NUM, 256)      # Output layer
        self.drop = nn.Dropout(0.2)                # Regularization
        self.act = nn.ReLU()                       # Activation
    
    def forward(self, rna):
        # Process gene expression data
        rna = F.relu(self.line(rna))
        rna = F.relu(self.line2(rna))
        return rna
class kaggle(nn.Module):
    def __init__(self, in_features, hidden_features=256, act_layer=nn.GELU, drop=0.1, pred=True):
        super().__init__()
        #out_features = out_features or in_features
        hidden_features = hidden_features
        self.q = nn.Linear(in_features, in_features)
        self.k = nn.Linear(in_features, in_features)
        self.v = nn.Linear(in_features, in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.pred = pred
        if pred==True:
            self.fc2 = nn.Linear(hidden_features,256)
        else:
            self.fc2 =  nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x0 = x
        q = self.q(x).unsqueeze(2)
        k = self.k(x).unsqueeze(2)
        v = self.v(x).unsqueeze(2)
        attn = (q @ k.transpose(-2, -1))
        #print(attn.size())
        attn = attn.softmax(dim=-1)
        x = (attn @ v).squeeze(2)
        #print(x.size())
        x += x0
        x1 = x
        x = self.fc1(x)
        x = self.act(x)
        #x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        # x = self.drop(x)
        if self.pred==False:
            x += x1
        x = x.squeeze(0)
        return x
# Drug feature dimension
drug_dim = 256

# Classifier Network (combines drug and gene features)
class Classifier(nn.Sequential):
    def __init__(self, model_drug, model_gene):
        super(Classifier, self).__init__()
        # Initialize feature extractors
        self.model_drug = model_drug  # Drug feature model
        self.model_gene = model_gene  # Gene feature model
        self.dropout = nn.Dropout(0.2)  # Regularization
        
        # Define predictor network dimensions
        self.hidden_dims = [1024, 1024, 512]
        layer_size = len(self.hidden_dims) + 1
        # Input dimension: drug_dim + gene_output_dim (256)
        dims = [drug_dim + 256] + self.hidden_dims + [1]  
        # Create fully connected layers
        self.predictor = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)]
        )
    
    def forward(self, G1, G2, rna, func):
        # Extract drug features
        v_D = self.model_drug(G1, G2, func)
        # Extract gene features
        v_P = self.model_gene(rna)
        # Concatenate features
        v_f = torch.cat((v_D, v_P), 1)
        
        # Pass through predictor network
        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):  # Output layer (no activation)
                v_f = l(v_f)
            else:  # Hidden layers (ReLU + dropout)
                v_f = F.relu(self.dropout(l(v_f)))
        return v_f

# Import GIN-based drug feature extractor
from graphconv_drug import GINConvNet

# Main Model Class (DVFGCONV)
class DVFGCONV:
    def __init__(self, modeldir):
        # Initialize submodels
        model_drug = GINConvNet(output_dim=drug_dim)  # Drug feature extractor
        model_gene = MLP1()                           # Gene feature extractor
        # Integrate into classifier
        self.model = Classifier(model_drug, model_gene)  
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.modeldir = modeldir
        # Output files
        self.record_file = os.path.join(self.modeldir, "valid_markdowntable.txt")
        self.pkl_file = os.path.join(self.modeldir, "loss_curve_iter.pkl")
    
    # Model Evaluation Method
    def test(self, datagenerator, model):
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
            label = y.view(-1).float().to(device)
            
            # Calculate loss
            loss_fct = torch.nn.MSELoss()
            n = torch.squeeze(score, 1).float()
            loss = loss_fct(n, label)
            
            # Collect predictions
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            y_label.extend(label_ids.flatten().tolist())
            y_pred.extend(logits.flatten().tolist())
        
        model.train()  # Restore training mode
        
        # Calculate evaluation metrics
        mse = mean_squared_error(y_label, y_pred)
        rmse = np.sqrt(mse)
        pearson_corr, pearson_p = pearsonr(y_label, y_pred)
        spearman_corr, spearman_p = spearmanr(y_label, y_pred)
        ci = concordance_index(y_label, y_pred)
        
        return y_label, y_pred, mse, rmse, pearson_corr, pearson_p, \
               spearman_corr, spearman_p, ci, loss
    
    # Model Training Method
    def train(self, train_drug, val_drug):
        # Hyperparameters
        lr = 0.0001        # Initial learning rate
        decay = 0.0004      # Weight decay
        BATCH_SIZE = 128    # Batch size
        train_epoch = 400   # Training epochs
        
        # Device placement
        self.model = self.model.to(self.device)
        # Optimizer
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
        loss_history = []  # Loss tracking
        
        # Data loaders
        training_generator = DataLoader(
            train_drug, 
            collate_fn=my_collate, 
            batch_size=BATCH_SIZE, 
            shuffle=True
        )
        validation_generator = DataLoader(
            val_drug, 
            collate_fn=my_collate, 
            batch_size=BATCH_SIZE, 
            shuffle=False
        )
        
        # Initialize best model tracking
        max_MSE = 10000
        valid_metric_record = []  # Validation metrics storage
        # Table headers for reporting
        valid_metric_header = [
            '# epoch', "MSE", 'RMSE', "Pearson Correlation", "with p-value", 
            'Spearman Correlation', "with p-value2", "Concordance Index"
        ]
        table = PrettyTable(valid_metric_header)
        # Float formatting function
        float2str = lambda x: '%0.4f' % x  
        
        # Training initialization
        print('--- Starting Training ---')
        # TensorBoard logger
        writer = SummaryWriter(self.modeldir, comment='Drug_Transformer_MLP')
        t_start = time.time()
        iteration_loss = 0  # Global iteration counter
        
        # Epoch loop
        for epo in range(train_epoch):
            loss_sum = 0  # Epoch loss accumulator
            
            # Learning rate scheduling
            if epo == 100:
                opt = torch.optim.Adam(self.model.parameters(), lr=0.00005, weight_decay=decay)
            elif epo == 200:
                opt = torch.optim.Adam(self.model.parameters(), lr=0.00003, weight_decay=decay)
            elif epo == 300:
                opt = torch.optim.Adam(self.model.parameters(), lr=0.00002, weight_decay=decay)
            elif epo == 350:
                opt = torch.optim.Adam(self.model.parameters(), lr=0.00001, weight_decay=decay)
            
            # Batch iteration
            for batch_idx, (y, g1, g2, rna, func) in enumerate(training_generator):
                # Forward pass
                score = self.model(
                    g1.to(device), 
                    g2.to(device) if g2 else None, 
                    rna.to(device), 
                    func.to(device)
                )
                
                # Loss calculation
                label = y.to(device)
                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(score, 1).float()
                loss = loss_fct(n, label)
                
                # Record and log loss
                loss_history.append(loss.item())
                writer.add_scalar("Loss/train", loss.item(), iteration_loss)
                iteration_loss += 1
                loss_sum += loss.item()
                
                # Backpropagation
                opt.zero_grad()
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
                
                # Update best model
                if mse < max_MSE:
                    max_MSE = mse
                    print(f'New best model at epoch {epo+1} with MSE: {mse:.4f}')
                    
                    # Log validation metrics
                    writer.add_scalar("valid/mse", mse, epo)
                    writer.add_scalar('valid/rmse', rmse, epo)
                    writer.add_scalar("valid/pearson", person, epo)
                    writer.add_scalar("valid/concordance_index", CI, epo)
                    writer.add_scalar("valid/spearman", spearman, epo)
                    writer.add_scalar("Loss/valid", loss_val.item(), iteration_loss)
            
            # Add to results table
            table.add_row(lst)
        
        # Save training artifacts
        with open(self.record_file, 'w') as fp:
            fp.write(table.get_string())
        with open(self.pkl_file, 'wb') as pck:
            pickle.dump(loss_history, pck)
        
        # Finalize training
        duration = time.time() - t_start
        print(f'--- Training Completed in {duration:.2f} seconds ---')
        writer.flush()
        writer.close()
    
    # Model Saving Method
    def save_model(self):
        save_path = os.path.join(self.modeldir, 'model.pt')
        torch.save(self.model.state_dict(), save_path)
        print(f'Model saved to {save_path}')
    
    # Model Loading Method
    def load_pretrained(self, path):
        # Create directory if needed
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Device-compatible loading
        if self.device.type == 'cuda':
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
        
        # Handle DataParallel naming
        if next(iter(state_dict)).startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # Load state dict
        self.model.load_state_dict(state_dict)
        print(f'Pre-trained model loaded from {path}')


if __name__ == '__main__':

    #  split data
    from Step2_DataEncoding import DataEncoding
    # #from Step2_DataEncoding_cmpot import DataEncoding
    vocab_dir = '/home/s3540/DTI/DVFGCONV-main'
    obj = DataEncoding(vocab_dir=vocab_dir)
    #
    # save data 
    # traindata, testdata = obj.Getdata.ByCancer(random_seed=1)
    # obj.encode(
    #  traindata=traindata,
    #  testdata=testdata)
    modeldir = 'Model_80'
    modelfile = modeldir + '/model.pt'
    #load model
    net = DVFGCONV(modeldir=modeldir)
    traindata = MolData('train')
    testdata = MolData('test')
    net.train(train_drug=traindata,val_drug=testdata)
    #save model
    net.save_model()
    print("Model Saveed :{}".format(modelfile))



