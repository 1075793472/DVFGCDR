import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx


def atom_features(atom):
    """
    Generate feature vector for an atom using one-hot encoding for various properties.
    
    Args:
        atom (rdkit.Chem.Atom): The atom object to extract features from
        
    Returns:
        np.array: Feature vector containing:
            - Atom symbol (one-hot encoded)
            - Degree (one-hot encoded)
            - Total number of hydrogens (one-hot encoded)
            - Implicit valence (one-hot encoded)
            - Aromaticity flag (binary)
    """    
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'Pd', 'Co',  'H',
                                           'Pt', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])
#
# def atom_features(atom):
#     possible_atom =  ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'Pd', 'Co',  'H',
#                                           'Pt', 'Unknown'] #DU代表其他原子
#     atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atom)
#     atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(),[0, 1, 2, 3, 4, 5, 6, 7, 8,9,10])
#     atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
#     atom_features += one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8,9,10])
#     #atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 1])
#     atom_features +=one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8,9,10])
#     atom_features += one_of_k_encoding_unk(atom.GetHybridization(),
#                                            [Chem.rdchem.HybridizationType.SP,
#                                             Chem.rdchem.HybridizationType.SP2,
#                                             Chem.rdchem.HybridizationType.SP3,
#                                             Chem.rdchem.HybridizationType.SP3D])+[atom.GetIsAromatic()]
#     print(np.array(atom_features).shape)
#     return np.array(atom_features)
def one_of_k_encoding(x, allowable_set):
    """
    One-hot encoding for values in an allowable set.
    
    Args:
        x: Value to encode
        allowable_set (list): Set of allowable values
        
    Returns:
        list: One-hot encoded vector
    """
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """
    One-hot encoding for values in an allowable set, mapping unknowns to last element.
    
    Args:
        x: Value to encode
        allowable_set (list): Set of allowable values
        
    Returns:
        list: One-hot encoded vector
    """
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph1(smile):
    """
    Convert a SMILES string to a molecular graph representation.
    
    Args:
        smile (str): SMILES string representing the molecule
        
    Returns:
        tuple: Contains:
            c_size (int): Number of atoms in the molecule
            features (list): Atom features for each atom
            edge_index (torch.LongTensor): Edge connections [2 x num_edges]
            edge_attr (torch.FloatTensor): Edge attributes [num_edges]
            W_spr (tuple): Sparse distance matrix (indices, values, shape)
    """
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    # edges_type=[]
    for bond in mol.GetBonds():
        bond_type = get_bond_features(bond)
        bond_type = np.argmax(bond_type)
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type])
        # edges_type.append(bond_type)
    g = nx.Graph()
    g.add_weighted_edges_from(edges)
    g = g.to_undirected()
    edgewidth = []
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        a = list(g.get_edge_data(e1, e2).values())[0]
        edgewidth.append(a)
    return c_size, features, edge_index,edgewidth
def bond_to_edge(bond):
    src = bond.GetBeginAtomIdx()
    dst = bond.GetEndAtomIdx()
    bond_type = bond.GetBondTypeAsDouble()
    edge = [src, dst, bond_type]
    return edge
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
from torch_geometric.utils import dense_to_sparse
def atom_to_node(atom):
    idx = atom.GetIdx()
    symbol = atom.GetSymbol()
    atom_nb = atom.GetAtomicNum()
    formal_charge = atom.GetFormalCharge()
    implicit_valence = atom.GetImplicitValence()
    ring_atom = atom.IsInRing()
    degree = atom.GetDegree()
    hybridization = atom.GetHybridization()
    # print(idx, symbol, atom_nb)
    # print(ring_atom)
    # print(degree)
    # print(hybridization)
    node = [idx, atom_nb, formal_charge, implicit_valence, ring_atom]
    return node

def is_sorted(l):
    return all(l[i] <= l[i+1] for i in range(len(l)-1))
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    # nodes = []
    # for atom in mol.GetAtoms():
    #     nodes.append(atom_to_node(atom))
    # idx = [n[0] for n in nodes]
    # assert is_sorted(idx)
    # nodes = np.array(nodes, dtype=float)[:,1:]
    edges = []
    for bond in mol.GetBonds():
        edges.append(bond_to_edge(bond))
    # for bond in mol.GetBonds():
    #     edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    #g = nx.Graph(edges).to_directed()
    # edge_index = []
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    edge_index = torch.LongTensor([src, dst])
    edge_attr = torch.FloatTensor([e[2] for e in edges])
    # for e1, e2 in g.edges:
    #     edge_index.append([e1, e2])
    z=torch.LongTensor(edge_index)
    try:
        if AllChem.EmbedMolecule(mol, randomSeed=0xf00d) == -1:  # optional random seed for reproducibility)
            AllChem.Compute2DCoords(mol)

        with np.errstate(divide='ignore'):
            W = 1. / Chem.rdmolops.Get3DDistanceMatrix(mol)
        W[np.isinf(W)] = 0
    except Exception as e:
        try:
            # TODO (Yulun): make this the first try?
            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol, randomSeed=0xf00d) == -1:  # optional random seed for reproducibility)
                AllChem.Compute2DCoords(mol)
            mol = Chem.RemoveHs(mol)

            with np.errstate(divide='ignore'):
                W = 1. / Chem.rdmolops.Get3DDistanceMatrix(mol)
            W[np.isinf(W)] = 0
        except Exception:
            num_atoms = mol.GetNumAtoms()
            W = np.zeros((num_atoms, num_atoms))
        # preserve top ratio*n entries
    threshold = np.sort(W, axis=None)[::-1][min(int(2 * len(W)) + 1, len(W) ** 2) - 1]
    W[W < threshold] = 0
    # convert to sparse representation
    W_spr = dense_to_sparse(torch.FloatTensor(W))
    return c_size, features, edge_index,edge_attr,W_spr
def get_bond_features(bond):
    """
    Generate feature vector for a bond (not used in main function but kept for reference).
    
    Args:
        bond (rdkit.Chem.Bond): Bond object
        
    Returns:
        np.array: Bond features vector containing:
            - Bond type flags (single, double, triple, aromatic)
            - Conjugated flag
            - Ring membership flag
    """
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(bond_feats)
#
# def seq_cat(prot):
#     x = np.zeros(max_seq_len)
#     for i, ch in enumerate(prot[:max_seq_len]):
#         x[i] = seq_dict[ch]
#     return x


