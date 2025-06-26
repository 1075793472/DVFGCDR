# python3
# -*- coding:utf-8 -*-

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file：PycharmProject-PyCharm-DataEncoding.py
@time:2021/9/7 10:04 
"""

def SparseTensor(row, col, value, sparse_sizes):
    return torch.sparse_coo_tensor(indices=torch.stack([row, col]),
                                   values=value,
                                   size=sparse_sizes)
def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud
import os
import numpy as np
import pandas as pd
import codecs
from subword_nmt.apply_bpe import BPE
from Step1_getData import GetData
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
from sklearn import preprocessing
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
def save_drug(xd,smile_graph):
    data_list = []
    data_len = len(xd)
    for i in range(data_len):
        print('Converting SMILES to graph: {}/{}'.format(i + 1, data_len))
        smiles = xd[i]
        # # convert SMILES to molecular representation using rdkit
        # c_size, features, edge_index = smile_graph[smiles]
        # # make the graph ready for PyTorch Geometrics GCN algorithms:
        # GCNData = DATA.Data(x=torch.Tensor(features),
        #                     edge_index=torch.LongTensor(edge_index).transpose(1, 0))
        data_list.append(smile_graph[smiles])
    np.save('test_drug.npy', data_list)

    return data_list
from torch_geometric.nn import GCNConv, radius_graph,knn,knn_graph
class TestbedDataset():
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xrna=None, y=None, transform=None,
                 pre_transform=None, smile_graph1=None,smile_graph2=None):

        # root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__()
        self.dataset = dataset
        self.process(xd, xrna, smile_graph1,smile_graph2, y,dataset)
    def process(self, xd,xrna,smile_graph1,smile_graph2,labels,dataset):
        data_list = []
        data_list1 = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i + 1, data_len))
            smiles = xd[i]
            #c_size, features, edge_index, edge_arr = smile_graph[smiles]

            # convert SMILES to molecular representation using rdkit
            #c_size, features, edge_index,edge_arr,g2 = smile_graph[smiles]
            c_size, features, edge_index, edge_arr,edge_type, g2 ,bong_edge,angle,atompos,fun= smile_graph1[smiles]
            c_size2, features2, edge_index2, edge_arr2, edge_type2, g22, bong_edge2, angle2, atompos2,fun = smile_graph2[smiles]

           # features=features/features.shape[]
           # features2 = features2 /5

            #atompos2 = pc_normalize(atompos2)
            edge_index2=radius_graph(torch.FloatTensor(atompos2),r=5,max_num_neighbors=35)
            #edge_index2 =knn(torch.FloatTensor(atompos2),torch.FloatTensor(atompos2),15)
            #edge_index2 = knn_graph(torch.FloatTensor(atompos2), 12,loop=False)
            # edge_index3 = radius_graph(torch.FloatTensor(atompos), r=6)
            # edge_index3=np.asarray(edge_index3.T,order='C')
            # A1= edge_index3.view([('', edge_index3.dtype)]*edge_index3.shape[1])
            # A2= edge_index.view([('', edge_index.dtype)]*edge_index.shape[1])
            #edge_index4=np.setdiff1d(A1,A2).view(edge_index3.dtype).reshape(-1,edge_index3.shape[1])
            # edge_index = torch.LongTensor(edge_index).transpose(1, 0)
            # edge_index4=torch.LongTensor(edge_index4).transpose(1,0)

            # edge_index5=torch.cat((edge_index,edge_index4),1)
            # zzz=np.repeat(np.array([[5,3]]),edge_index4.shape[1],axis=0)
            # edge_arr4=np.vstack((edge_arr,zzz))
            atompos2 = pc_normalize(atompos2)
            atompos = pc_normalize(atompos)
            # if dataset=='_train':
            #  atompos2=translate_pointcloud(atompos2)
            #features=np.expand_dims(features,axis=-1)
            features=np.hstack((features,atompos))
            features2 = np.hstack((features2, atompos2))
            min_max=preprocessing.MinMaxScaler()
            edge_type1=min_max.fit_transform(edge_type.reshape(-1,1))
            edge_type1=edge_type1.reshape(-1)
            result=[]
            X_MAX=np.max(edge_type2.reshape(-1,1))
            X_MIN=np.min(edge_type2.reshape(-1,1))
            for j in edge_type2.reshape(-1,1):
                result.append((float(j[0]-X_MIN)/(X_MAX-X_MIN)))
            result=np.array(result,dtype=float)
            result=result.reshape(-1,1)
            fun=min_max.fit_transform(fun.reshape(-1,1))
            fun = fun.reshape(-1)
            #features = min_max.fit_transform(features)
           # edge_type1[edge_type1<0.2]=0
            angle=min_max.fit_transform(angle.reshape(-1,1))
            #angle=1-angle.reshape(-1)
            #bot_col=edge_arr[:,[2,3,4,13,24]]
            geo_f=np.hstack((edge_arr,edge_type.reshape(-1,1)))
            #geo_f = edge_arr.reshape(-1, 1)
            geo_f2 =np.hstack((edge_arr2,edge_type2.reshape(-1,1)))
            # GCNData = DATA.Data(x=torch.LongTensor(features),
            #                     edge_attr=torch.FloatTensor(geo_f),
            #                     edge_index=torch.LongTensor(edge_index).transpose(1,0),
            #                     y=torch.FloatTensor([labels[i]])
            #                     )
            GCNData = DATA.Data(x=torch.LongTensor(features),
                                edge_attr=torch.LongTensor(edge_arr),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels[i]])
                                )
            # GCNData = DATA.Data(x=torch.LongTensor(features),
            #                     edge_attr=torch.LongTensor(edge_arr4),
            #                     edge_index=edge_index5,
            #                     y=torch.FloatTensor([labels[i]]))
            # GCNData1 = DATA.Data(x=torch.FloatTensor(features),
            #                     edge_attr=g2[1],
            #                     edge_index=g2[0])
            # GCNData1 = DATA.Data(x=torch.FloatTensor(atompos2),
            #                      edge_attr=torch.FloatTensor(result),
            #                      edge_index=torch.LongTensor(edge_index2).transpose(1,0))
            GCNData1 = DATA.Data(x=torch.FloatTensor(atompos2),
                                 edge_attr=torch.FloatTensor(result),
                                 edge_index=edge_index2)
            GCNData.target = torch.LongTensor([xrna[i]])
            GCNData.func=fun
            data_list.append(GCNData)
            data_list1.append(GCNData1)
        print('Graph construction done. Saving to file.')
        # save preprocessed data:
        torch.save(data_list, 'g1'+self.dataset+'.pt')
        torch.save(data_list1,'g2'+self.dataset+'.pt')
class DataEncoding:
    def __init__(self,vocab_dir):
        self.vocab_dir = vocab_dir
        # 一个获取数据的函数类
        self.Getdata = GetData()

    def _drug2emb_encoder(self,smile):
        vocab_path = "{}/drug_codes_chembl_freq_1500.txt".format(self.vocab_dir)
        sub_csv = pd.read_csv("{}/subword_units_map_chembl_freq_1500.csv".format(self.vocab_dir))

        bpe_codes_drug = codecs.open(vocab_path)
        dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

        idx2word_d = sub_csv['index'].values
        words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

        max_d = 50
        t1 = dbpe.process_line(str(smile)).split()  # split
        try:
            i1 = np.asarray([words2idx_d[i] for i in t1])  # index
        except:
            i1 = np.array([0])

        l = len(i1)
        if l < max_d:
            i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
            input_mask = ([1] * l) + ([0] * (max_d - l))
        else:
            i = i1[:max_d]
            input_mask = [1] * max_d

        return i, np.asarray(input_mask)

    def encode(self,traindata,testdata):
        traindata = traindata.reset_index()
        #traindata['Label'] = traindata['Binary_IC50']
        traindata['Label'] = traindata['LN_IC50']
        testdata = testdata.reset_index()
        #testdata['Label'] = testdata['Binary_IC50']
        testdata['Label'] = testdata['LN_IC50']
        drug_smiles=pd.read_csv('smile_inchi.csv',index_col=0)
        drugid2smile = dict(zip(drug_smiles['drug_id'], drug_smiles['smiles']))
        train_drugs = np.array([drugid2smile[i] for i in traindata['DRUG_ID']])
        test_drugs = np.array([drugid2smile[i] for i in testdata['DRUG_ID']])
        smile_graph1 = np.load('graph2d1.npy', allow_pickle=True).item()
        smile_graph2 = np.load('graph3d1.npy', allow_pickle=True).item()
        # train_rnadata, test_rnadata = self.Getdata.getRna(
        #     traindata=traindata,
        #     testdata=testdata)
        train_rnadata, test_rnadata= self.Getdata.getRna(
            traindata=traindata,
            testdata=testdata)
        trainDRUG = TestbedDataset(root='data', dataset='_train', xd=train_drugs,
                                    smile_graph1=smile_graph1,smile_graph2=smile_graph2,y=traindata['Label'].values,xrna=train_rnadata)
        testDRUG = TestbedDataset(root='data', dataset='_test', xd=test_drugs,
                                   smile_graph1=smile_graph1,smile_graph2=smile_graph2,y=testdata['Label'].values,xrna=test_rnadata)
        return None
    def encode_miss(self,traindata,testdata):
        traindata = traindata.reset_index()
        #traindata['Label'] = traindata['Binary_IC50']
        traindata['Label'] = traindata['LN_IC50']
        testdata = testdata.reset_index()
        #testdata['Label'] = testdata['Binary_IC50']
        testdata['Label'] = 1
        drug_smiles=pd.read_csv('smile_inchi.csv',index_col=0)
        drugid2smile = dict(zip(drug_smiles['drug_id'], drug_smiles['smiles']))
        train_drugs = np.array([drugid2smile[i] for i in traindata['DRUG_ID']])
        test_drugs = np.array([drugid2smile[i] for i in testdata['DRUG_ID']])
        smile_graph1 = np.load('graph2d1.npy', allow_pickle=True).item()
        smile_graph2 = np.load('graph3d1.npy', allow_pickle=True).item()
        # train_rnadata, test_rnadata = self.Getdata.getRna(
        #     traindata=traindata,
        #     testdata=testdata)
        train_rnadata, test_rnadata= self.Getdata.getRna(
            traindata=traindata,
            testdata=testdata)
        trainDRUG = TestbedDataset(root='data', dataset='_train', xd=train_drugs,
                                    smile_graph1=smile_graph1,smile_graph2=smile_graph2,y=traindata['Label'].values,xrna=train_rnadata)
        testDRUG = TestbedDataset(root='data', dataset='_test', xd=test_drugs,
                                   smile_graph1=smile_graph1,smile_graph2=smile_graph2,y=testdata['Label'].values,xrna=test_rnadata)
        return None
if __name__ == '__main__':
    vocab_dir = '/home/s3540/DTI/DFVGCONV-main'
    obj = DataEncoding(vocab_dir=vocab_dir)
    traindata, testdata = obj.Getdata.ByCancer(random_seed= 1)

    traindata, train_rnadata, testdata, test_rnadata = obj.encode(
        traindata=traindata,
        testdata=testdata)

    print(traindata, train_rnadata, testdata, test_rnadata)