import math
import torch
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
def one_hot_vector(val, lst, add_unknown=True):
    if add_unknown:
        vec = np.zeros(len(lst) + 1)
    else:
        vec = np.zeros(len(lst))

    vec[lst.index(val) if val in lst else -1] = 1
    return vec

def pad_array(array, shape):
    padded_array = np.zeros(shape, dtype=np.float)
    if len(shape) == 2:
        padded_array[:array.shape[0], :array.shape[1]] = array
    elif len(shape) == 3:
        padded_array[:array.shape[0], :array.shape[1], :] = array
    return padded_array

def get_atom_features(atom, d_atom):
    # 100+1=101 dimensions
    v1 = one_hot_vector(atom.GetAtomicNum(), [i for i in range(1, 101)])

    # 5+1=6 dimensions
    v2 = one_hot_vector(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                  Chem.rdchem.HybridizationType.SP2,
                                                  Chem.rdchem.HybridizationType.SP3,
                                                  Chem.rdchem.HybridizationType.SP3D,
                                                  Chem.rdchem.HybridizationType.SP3D2])

    # 8 dimensions
    v3 = [
        atom.GetTotalNumHs(includeNeighbors=True) / 8,
        atom.GetDegree() / 4,
        atom.GetFormalCharge() / 8,
        atom.GetTotalValence() / 8,
        0 if math.isnan(atom.GetDoubleProp('_GasteigerCharge')) or math.isinf(
            atom.GetDoubleProp('_GasteigerCharge')) else atom.GetDoubleProp('_GasteigerCharge'),
        0 if math.isnan(atom.GetDoubleProp('_GasteigerHCharge')) or math.isinf(
            atom.GetDoubleProp('_GasteigerHCharge')) else atom.GetDoubleProp('_GasteigerHCharge'),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing())
    ]

    # index for position encoding
    v4 = [
        atom.GetIdx() + 1  # start from 1
    ]

    attributes = np.concatenate([v1, v2, v3, v4], axis=0)

    # total for 32 dimensions
    assert len(attributes) == d_atom + 1
    return attributes


def get_bond_features(bond, d_edge):
    # 4 dimensions
    v1 = one_hot_vector(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE,
                                             Chem.rdchem.BondType.DOUBLE,
                                             Chem.rdchem.BondType.TRIPLE,
                                             Chem.rdchem.BondType.AROMATIC], add_unknown=False)

    # 6 dimensions
    v2 = one_hot_vector(bond.GetStereo(), [Chem.rdchem.BondStereo.STEREOANY,
                                           Chem.rdchem.BondStereo.STEREOCIS,
                                           Chem.rdchem.BondStereo.STEREOE,
                                           Chem.rdchem.BondStereo.STEREONONE,
                                           Chem.rdchem.BondStereo.STEREOTRANS,
                                           Chem.rdchem.BondStereo.STEREOZ], add_unknown=False)

    # 3 dimensions
    v3 = [
        int(bond.GetIsConjugated()),
        int(bond.GetIsAromatic()),
        int(bond.IsInRing())
    ]

    # total for 115+13=128 dimensions
    attributes = np.concatenate([v1, v2, v3])

    assert len(attributes) == d_edge
    return attributes

def load_data_from_mol(mol, d_atom, d_edge, max_length):
    # Set Stereochemistry
    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
    Chem.rdmolops.AssignStereochemistryFrom3D(mol)
    AllChem.ComputeGasteigerCharges(mol)

    # Get Node features Init
    node_features = np.array([get_atom_features(atom, d_atom) for atom in mol.GetAtoms()])

    # Get Bond features
    bond_features = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms(), d_edge))

    for bond in mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtom().GetIdx()
        end_atom_idx = bond.GetEndAtom().GetIdx()
        bond_features[begin_atom_idx, end_atom_idx, :] = bond_features[end_atom_idx, begin_atom_idx, :] = get_bond_features(bond, d_edge)

    # Get Adjacency matrix without self loop
    adjacency_matrix = Chem.rdmolops.GetDistanceMatrix(mol).astype(np.float)

    # node_features.shape    = (num_atoms, d_atom) -> (max_length, d_atom)
    # bond_features.shape    = (num_atoms, num_atoms, d_edge) -> (max_length, max_length, d_edge)
    # adjacency_matrix.shape = (num_atoms, num_atoms) -> (max_length, max_length)
    return pad_array(node_features, (max_length, node_features.shape[-1])), \
           pad_array(bond_features, (max_length, max_length, bond_features.shape[-1])), \
           pad_array(adjacency_matrix, (max_length, max_length))
class Molecule:

    def __init__(self, mol, label, d_atom, d_edge, max_length):
        self.smile = Chem.MolToSmiles(mol)
        self.label = label
        self.node_features, self.bond_features, self.adjacency_matrix = load_data_from_mol(mol, d_atom, d_edge, max_length)

def construct_dataset(mol_list, label_list, d_atom, d_edge, max_length):
    output = [Molecule(mol, label, d_atom, d_edge, max_length)
              for (mol, label) in tqdm(zip(mol_list, label_list), total=len(mol_list))]

    return MolDataSet(output)
def construct_loader(mol_list, label_list, batch_size, d_atom, d_edge, max_length, shuffle=True):
    dataset = construct_dataset(mol_list, label_list, d_atom, d_edge, max_length)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=mol_collate_func, shuffle=shuffle,
                        drop_last=True, num_workers=0)
    return loader

if __name__ == "__main__":
    import pandas as pd

    # a = np.load('graph33d.npy', allow_pickle=True)
    # zz=a[0]
    # for i in  range(1,a.shape[0]):
    #     zz=np.vstack((zz,a[i]))
    #     #print('d')
    # zzz=np.arange(0,24).tolist()
    # z = np.where(~zz.any(axis=0))[0].tolist()
    #
    # a = np.load('graph33d.npy', allow_pickle=True).item()
    smile_graph = {}
    smile_graph_3d={}
    # smiles = "OCc1ccccc1CN"
    # smiles = r"[H]/[NH+]=C(\N)C1=CC(=O)/C(=C\C=c2ccc(=C(N)[NH3+])cc2)C=C1"
    # mol = AllChem.MolFromSmiles(smiles)
    # print(len(smiles))
    # print(mol)
    # data = mol_to_geognn_graph_data_MMFF3d(mol)
    max1=0
    atom_feature=[]
    edge_feature=[]
    df = pd.read_csv("smile_inchi.csv")
    compound_iso_smiles = []
    compound_iso_smiles += list(df['smiles'])
    compound_iso_smiles = set(compound_iso_smiles)
    z=[]
    # mol1 = AllChem.MolFromSmiles('OCc1ccccc1CN')
    # data = mol_to_geognn_graph_data_MMFF3d(mol1)
    atom_Num=[]
    degree=[]
    totol_hs=[]
    implicit_valence=[]
    formal_charge=[]
    hybridization=[]
    bond_type=[]
    bond_dir=[]
    chiral_tag=[]
    GetNumRadicalElectrons=[]
    distributions = np.zeros(11)
    max_length = -1
    for smile in compound_iso_smiles:
        mol = Chem.MolFromSmiles(smile)
        try:
            Chem.SanitizeMol(mol)
        except:
            continue
        atom_num = mol.GetNumAtoms() // 10
        if atom_num < 10:
            distributions[atom_num] += 1
        else:
            distributions[-1] += 1
        max_length = mol.GetNumAtoms() if mol.GetNumAtoms() > max_length else max_length
    #max_length = max([data.GetNumAtoms() for data in data_mol])
    d_atom = 115
    d_edge = 13
    for smile in compound_iso_smiles:
        mol = AllChem.MolFromSmiles(smile)
        node_features, bond_features,adjacency_matrix = load_data_from_mol(mol, d_atom, d_edge, max_length)
        smile_graph[smile] =  node_features, bond_features,adjacency_matrix
        print('d')
    np.save('../graph3d2.npy', smile_graph)
