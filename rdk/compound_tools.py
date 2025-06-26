#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
| Tools for compound features.
| Adapted from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/loader.py
"""
import os
from collections import OrderedDict

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
import deepchem as dc
from compound_constants import DAY_LIGHT_FG_SMARTS_LIST

# if name == 'atomic_num':
#     return atom.GetAtomicNum()
# elif name == 'chiral_tag':
#     return atom.GetChiralTag()
# elif name == 'degree':
#     return atom.GetDegree()
# elif name == 'explicit_valence':
#     return atom.GetExplicitValence()
# elif name == 'formal_charge':
#     return atom.GetFormalCharge()
# elif name == 'hybridization':
#     return atom.GetHybridization()
# elif name == 'implicit_valence':
#     return atom.GetImplicitValence()
# elif name == 'is_aromatic':
#     a = int(atom.GetIsAromatic())
#     return a
# elif name == 'mass':
#     return int(atom.GetMass())
# elif name == 'total_numHs':
#     return atom.GetTotalNumHs()
# elif name == 'num_radical_e':
#     return atom.GetNumRadicalElectrons()
# elif name == 'atom_is_in_ring':
#     return int(atom.IsInRing())
# elif name == 'valence_out_shell':
seq= ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Pt','Cl', 'Br', 'I', 'B', 'Pd', 'Co', 'H']
seq_dict={v:(i+1) for i,v in enumerate(seq)}
seq1= ['2','3','4','13']
seq_edge={v:(i+1) for i,v in enumerate(seq1)}
print('d')
def atom_features3(atom,
                explicit_H=True,
                use_chirality=False):
    xt = seq_dict[atom.GetSymbol()]
    return xt
def edge(edge,
                explicit_H=True,
                use_chirality=False):

    xt = [seq_edge[str(i)] for i in edge]
    return xt
def atom_features2(atom,
                explicit_H=True,
                use_chirality=False):
    a=atom.GetNumRadicalElectrons()
    print('d')
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
         'Yb', 'Sb', 'Sn', 'Ag', 'Pd',  'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
         'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])
    return np.array(results)
def atom_features1(mol):
    featurizer = dc.feat.graph_features.ConvMolFeaturizer()
    mol_object = featurizer.featurize(mol)
    features = mol_object[0].atom_features
    return  features
def atom_features(atom,
                explicit_H=True,
                use_chirality=False):
    a=atom.GetNumRadicalElectrons()
    print('d')
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
         'Yb', 'Sb', 'Sn', 'Ag', 'Pd',  'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
         'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])+ [atom.GetDegree()/10, atom.GetImplicitValence()/10,atom.GetTotalNumHs()/10]\
              + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
                ]) \
              + [atom.GetIsAromatic()]
    return np.array(results)
def atom_to_node(atom):
    idx = atom.GetIdx()
    symbol = atom.GetSymbol()
    atom_nb = atom.GetAtomicNum()
    formal_charge = atom.GetFormalCharge()
    hs=atom.GetTotalNumHs()
    implicit_valence = atom.GetImplicitValence()
    ring_atom = atom.IsInRing()
    degree = atom.GetDegree()
    hybridization = atom.GetHybridization()
    is_aromatic=atom.GetIsAromatic()
    # print(idx, symbol, atom_nb)
    # print(ring_atom)
    # print(degree)
    # print(hybridization)
    node = [idx, implicit_valence, hs, degree, is_aromatic]
    # node = [idx, atom_nb, formal_charge, implicit_valence, ring_atom,hs,degree,is_aromatic]
    return node

def get_gasteiger_partial_charges(mol, n_iter=12):
    """
    Calculates list of gasteiger partial charges for each atom in mol object.

    Args: 
        mol: rdkit mol object.
        n_iter(int): number of iterations. Default 12.

    Returns: 
        list of computed partial charges for each atom.
    """
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, nIter=n_iter,
                                                  throwOnParamFailure=True)
    partial_charges = [float(a.GetProp('_GasteigerCharge')) for a in
                       mol.GetAtoms()]
    return partial_charges


def create_standardized_mol_id(smiles):
    """
    Args:
        smiles: smiles sequence.

    Returns: 
        inchi.
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if not mol is None: # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles: # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return


def check_smiles_validity(smiles):
    """
    Check whether the smile can't be converted to rdkit mol object.
    """
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except Exception as e:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively.

    Args:
        mol: rdkit mol object.
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one.

    Args: 
        mol_list(list): a list of rdkit mol object.

    Returns:
        the largest mol.
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]

def rdchem_enum_to_list(values):
    """values = {0: rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, 
            1: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, 
            2: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, 
            3: rdkit.Chem.rdchem.ChiralType.CHI_OTHER}
    """
    return [values[i] for i in range(len(values))]


def safe_index(alist, elem):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return alist.index(elem)
    except ValueError:
        return len(alist) - 1


def get_atom_feature_dims(list_acquired_feature_names):
    """ tbd
    """
    return list(map(len, [CompoundKit.atom_vocab_dict[name] for name in list_acquired_feature_names]))


def get_bond_feature_dims(list_acquired_feature_names):
    """ tbd
    """
    list_bond_feat_dim = list(map(len, [CompoundKit.bond_vocab_dict[name] for name in list_acquired_feature_names]))
    # +1 for self loop edges
    return [_l + 1 for _l in list_bond_feat_dim]


class CompoundKit(object):
    """
    CompoundKit
    """
    atom_vocab_dict = {
        "atomic_num": list(range(1, 119)) + ['misc'],
        "chiral_tag": rdchem_enum_to_list(rdchem.ChiralType.values),
        "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "explicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "hybridization": rdchem_enum_to_list(rdchem.HybridizationType.values),
        "implicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "is_aromatic": [0, 1],
        "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'num_radical_e': [0, 1, 2, 3, 4, 'misc'],
        'atom_is_in_ring': [0, 1],
        'valence_out_shell': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size5': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size6': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size7': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size8': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    }
    bond_vocab_dict = {
        "bond_dir": rdchem_enum_to_list(rdchem.BondDir.values),
        "bond_type": rdchem_enum_to_list(rdchem.BondType.values),
        "is_in_ring": [0, 1],

        'bond_stereo': rdchem_enum_to_list(rdchem.BondStereo.values),
        'is_conjugated': [0, 1],
    }
    # float features
    atom_float_names = ["van_der_waals_radis", "partial_charge", 'mass']
    # bond_float_feats= ["bond_length", "bond_angle"]     # optional

    ### functional groups
    day_light_fg_smarts_list = DAY_LIGHT_FG_SMARTS_LIST
    day_light_fg_mo_list = [Chem.MolFromSmarts(smarts) for smarts in day_light_fg_smarts_list]

    morgan_fp_N = 200
    morgan2048_fp_N = 2048
    maccs_fp_N = 167

    period_table = Chem.GetPeriodicTable()

    ### atom

    @staticmethod
    def get_atom_value(atom, name):
        """get atom values"""
        if name == 'atomic_num':
            return atom.GetAtomicNum()
        elif name == 'chiral_tag':
            return atom.GetChiralTag()
        elif name == 'degree':
            return atom.GetDegree()
        elif name == 'explicit_valence':
            return atom.GetExplicitValence()
        elif name == 'formal_charge':
            return atom.GetFormalCharge()
        elif name == 'hybridization':
            return atom.GetHybridization()
        elif name == 'implicit_valence':
            return atom.GetImplicitValence()
        elif name == 'is_aromatic':
            a=int(atom.GetIsAromatic())
            return a
        elif name == 'mass':
            return int(atom.GetMass())
        elif name == 'total_numHs':
            return atom.GetTotalNumHs()
        elif name == 'num_radical_e':
            return atom.GetNumRadicalElectrons()
        elif name == 'atom_is_in_ring':
            return int(atom.IsInRing())
        elif name == 'valence_out_shell':
            return CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())
        else:
            raise ValueError(name)

    @staticmethod
    def get_atom_feature_id(atom, name):
        """get atom features id"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return safe_index(CompoundKit.atom_vocab_dict[name], CompoundKit.get_atom_value(atom, name))

    @staticmethod
    def get_atom_feature_size(name):
        """get atom features size"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return len(CompoundKit.atom_vocab_dict[name])

    ### bond

    @staticmethod
    def get_bond_value(bond, name):
        """get bond values"""
        if name == 'bond_dir':
            return bond.GetBondDir()
        elif name == 'bond_type':
            return bond.GetBondType()
        elif name == 'is_in_ring':
            return int(bond.IsInRing())
        elif name == 'is_conjugated':
            return int(bond.GetIsConjugated())
        elif name == 'bond_stereo':
            return bond.GetStereo()
        else:
            raise ValueError(name)

    @staticmethod
    def get_bond_feature_id(bond, name):
        """get bond features id"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return safe_index(CompoundKit.bond_vocab_dict[name], CompoundKit.get_bond_value(bond, name))

    @staticmethod
    def get_bond_feature_size(name):
        """get bond features size"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return len(CompoundKit.bond_vocab_dict[name])

    ### fingerprint

    @staticmethod
    def get_morgan_fingerprint(mol, radius=2):
        """get morgan fingerprint"""
        nBits = CompoundKit.morgan_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]
    
    @staticmethod
    def get_morgan2048_fingerprint(mol, radius=2):
        """get morgan2048 fingerprint"""
        nBits = CompoundKit.morgan2048_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_maccs_fingerprint(mol):
        """get maccs fingerprint"""
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]

    ### functional groups

    @staticmethod
    def get_daylight_functional_group_counts(mol):
        """get daylight functional group counts"""
        fg_counts = []
        for fg_mol in CompoundKit.day_light_fg_mo_list:
            sub_structs = Chem.Mol.GetSubstructMatches(mol, fg_mol, uniquify=True)
            fg_counts.append(len(sub_structs))
        return fg_counts

    @staticmethod
    def get_ring_size(mol):
        """return (N,6) list"""
        rings = mol.GetRingInfo()
        rings_info = []
        for r in rings.AtomRings():
            rings_info.append(r)
        ring_list = []
        for atom in mol.GetAtoms():
            atom_result = []
            for ringsize in range(3, 9):
                num_of_ring_at_ringsize = 0
                for r in rings_info:
                    if len(r) == ringsize and atom.GetIdx() in r:
                        num_of_ring_at_ringsize += 1
                if num_of_ring_at_ringsize > 8:
                    num_of_ring_at_ringsize = 9
                atom_result.append(num_of_ring_at_ringsize)
            
            ring_list.append(atom_result)
        return ring_list

    @staticmethod
    def atom_to_feat_vector(atom):
        """ tbd """
        atom_names = {
            "atomic_num": safe_index(CompoundKit.atom_vocab_dict["atomic_num"], atom.GetAtomicNum()),
            "chiral_tag": safe_index(CompoundKit.atom_vocab_dict["chiral_tag"], atom.GetChiralTag()),
            "degree": safe_index(CompoundKit.atom_vocab_dict["degree"], atom.GetTotalDegree()),
            "explicit_valence": safe_index(CompoundKit.atom_vocab_dict["explicit_valence"], atom.GetExplicitValence()),
            "formal_charge": safe_index(CompoundKit.atom_vocab_dict["formal_charge"], atom.GetFormalCharge()),
            "hybridization": safe_index(CompoundKit.atom_vocab_dict["hybridization"], atom.GetHybridization()),
            "implicit_valence": safe_index(CompoundKit.atom_vocab_dict["implicit_valence"], atom.GetImplicitValence()),
            "is_aromatic": safe_index(CompoundKit.atom_vocab_dict["is_aromatic"], int(atom.GetIsAromatic())),
            "total_numHs": safe_index(CompoundKit.atom_vocab_dict["total_numHs"], atom.GetTotalNumHs()),
            'num_radical_e': safe_index(CompoundKit.atom_vocab_dict['num_radical_e'], atom.GetNumRadicalElectrons()),
            'atom_is_in_ring': safe_index(CompoundKit.atom_vocab_dict['atom_is_in_ring'], int(atom.IsInRing())),
            'valence_out_shell': safe_index(CompoundKit.atom_vocab_dict['valence_out_shell'],
                                            CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())),
            'van_der_waals_radis': CompoundKit.period_table.GetRvdw(atom.GetAtomicNum()),
            'partial_charge': CompoundKit.check_partial_charge(atom),
            'mass': atom.GetMass(),
        }
        return atom_names

    @staticmethod
    def get_atom_names(mol):
        """get atom name list
        TODO: to be remove in the future
        """
        atom_features_dicts = []
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts.append(CompoundKit.atom_to_feat_vector(atom))

        ring_list = CompoundKit.get_ring_size(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts[i]['in_num_ring_with_size3'] = safe_index(
                    CompoundKit.atom_vocab_dict['in_num_ring_with_size3'], ring_list[i][0])
            atom_features_dicts[i]['in_num_ring_with_size4'] = safe_index(
                    CompoundKit.atom_vocab_dict['in_num_ring_with_size4'], ring_list[i][1])
            atom_features_dicts[i]['in_num_ring_with_size5'] = safe_index(
                    CompoundKit.atom_vocab_dict['in_num_ring_with_size5'], ring_list[i][2])
            atom_features_dicts[i]['in_num_ring_with_size6'] = safe_index(
                    CompoundKit.atom_vocab_dict['in_num_ring_with_size6'], ring_list[i][3])
            atom_features_dicts[i]['in_num_ring_with_size7'] = safe_index(
                    CompoundKit.atom_vocab_dict['in_num_ring_with_size7'], ring_list[i][4])
            atom_features_dicts[i]['in_num_ring_with_size8'] = safe_index(
                    CompoundKit.atom_vocab_dict['in_num_ring_with_size8'], ring_list[i][5])

        return atom_features_dicts
        
    @staticmethod
    def check_partial_charge(atom):
        """tbd"""
        pc = atom.GetDoubleProp('_GasteigerCharge')
        if pc != pc:
            # unsupported atom, replace nan with 0
            pc = 0
        if pc == float('inf'):
            # max 4 for other atoms, set to 10 here if inf is get
            pc = 10
        return pc


class Compound3DKit(object):
    """the 3Dkit of Compound"""
    @staticmethod
    def get_atom_poses(mol, conf):
        """tbd"""
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses

    @staticmethod
    def get_MMFF_atom_poses(mol, numConfs=None, return_energy=False):
        """the atoms of mol will be changed in some cases."""
        try:
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
            ### MMFF generates multiple conformations
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            index = np.argmin([x[1] for x in res])
            energy = res[index][1]
            conf = new_mol.GetConformer(id=int(index))
        except:
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)
            energy = 0
            conf = new_mol.GetConformer()

        atom_poses = Compound3DKit.get_atom_poses(new_mol, conf)
        if return_energy:
            return new_mol, atom_poses, energy
        else:
            return new_mol, atom_poses

    @staticmethod
    def get_2d_atom_poses(mol):
        """get 2d atom poses"""
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return atom_poses

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        """get bond lengths"""
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))
        bond_lengths = np.array(bond_lengths, 'float32')
        return bond_lengths

    @staticmethod
    def get_superedge_angles(edges, atom_poses, dir_type='HT'):
        """get superedge angles"""
        def _get_vec(atom_poses, edge):
            return atom_poses[edge[1]] - atom_poses[edge[0]]
        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle

        E = len(edges)
        edge_indices = np.arange(E)
        super_edges = []
        bond_angles = []
        bond_angle_dirs = []
        for tar_edge_i in range(E):
            tar_edge = edges[tar_edge_i]
            if dir_type == 'HT':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]
            elif dir_type == 'HH':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[1]]
            else:
                raise ValueError(dir_type)
            for src_edge_i in src_edge_indices:
                if src_edge_i == tar_edge_i:
                    continue
                src_edge = edges[src_edge_i]
                src_vec = _get_vec(atom_poses, src_edge)
                tar_vec = _get_vec(atom_poses, tar_edge)
                super_edges.append([src_edge_i, tar_edge_i])
                angle = _get_angle(src_vec, tar_vec)
                bond_angles.append(angle)
                bond_angle_dirs.append(src_edge[1] == tar_edge[0])  # H -> H or H -> T

        if len(super_edges) == 0:
            super_edges = np.zeros([0, 2], 'int64')
            bond_angles = np.zeros([0,], 'float32')
        else:
            super_edges = np.array(super_edges, 'int64')
            bond_angles = np.array(bond_angles, 'float32')
        return super_edges, bond_angles, bond_angle_dirs



def new_smiles_to_graph_data(smiles, **kwargs):
    """
    Convert smiles to graph data.
    """
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        return None
    data = new_mol_to_graph_data(mol)
    return data


def new_mol_to_graph_data(mol):
    """
    mol_to_graph_data

    Args:
        atom_features: Atom features.
        edge_features: Edge features.
        morgan_fingerprint: Morgan fingerprint.
        functional_groups: Functional groups.
    """
    if len(mol.GetAtoms()) == 0:
        return None

    atom_id_names = list(CompoundKit.atom_vocab_dict.keys()) + CompoundKit.atom_float_names
    bond_id_names = list(CompoundKit.bond_vocab_dict.keys())

    data = {}

    ### atom features
    data = {name: [] for name in atom_id_names}

    raw_atom_feat_dicts = CompoundKit.get_atom_names(mol)
    for atom_feat in raw_atom_feat_dicts:
        for name in atom_id_names:
            data[name].append(atom_feat[name])

    ### bond and bond features
    for name in bond_id_names:
        data[name] = []
    data['edges'] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j and j->i
        data['edges'] += [(i, j), (j, i)]
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name)
            data[name] += [bond_feature_id] * 2

    #### self loop
    N = len(data[atom_id_names[0]])
    for i in range(N):
        data['edges'] += [(i, i)]
    for name in bond_id_names:
        bond_feature_id = get_bond_feature_dims([name])[0] - 1   # self loop: value = len - 1
        data[name] += [bond_feature_id] * N

    ### make ndarray and check length
    for name in list(CompoundKit.atom_vocab_dict.keys()):
        data[name] = np.array(data[name], 'int64')
    for name in CompoundKit.atom_float_names:
        data[name] = np.array(data[name], 'float32')
    for name in bond_id_names:
        data[name] = np.array(data[name], 'int64')
    data['edges'] = np.array(data['edges'], 'int64')

    ### morgan fingerprint
    data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), 'int64')
    # data['morgan2048_fp'] = np.array(CompoundKit.get_morgan2048_fingerprint(mol), 'int64')
    data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), 'int64')
    data['daylight_fg_counts'] = np.array(CompoundKit.get_daylight_functional_group_counts(mol), 'int64')
    return data


def mol_to_graph_data(mol):
    """
    mol_to_graph_data

    Args:
        atom_features: Atom features.
        edge_features: Edge features.
        morgan_fingerprint: Morgan fingerprint.
        functional_groups: Functional groups.
    """
    if len(mol.GetAtoms()) == 0:
        return None

    atom_id_names = [
        "atomic_num", "chiral_tag", "degree", "explicit_valence", 
        "formal_charge", "hybridization", "implicit_valence", 
        "is_aromatic", "total_numHs",
    ]
    bond_id_names = [
        "bond_dir", "bond_type", "is_in_ring",
    ]
    
    data = {}
    for name in atom_id_names:
        data[name] = []
    data['mass'] = []
    for name in bond_id_names:
        data[name] = []
    data['edges'] = []

    ### atom features
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return None
        for name in atom_id_names:
            data[name].append(CompoundKit.get_atom_feature_id(atom, name) + 1)  # 0: OOV
        data['mass'].append(CompoundKit.get_atom_value(atom, 'mass') * 0.01)

    ### bond features
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j and j->i
        data['edges'] += [(i, j), (j, i)]
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name) + 1   # 0: OOV
            data[name] += [bond_feature_id] * 2
            # print('dd')

    ### self loop (+2)
    # N = len(data[atom_id_names[0]])
    # for i in range(N):
    #     data['edges'] += [(i, i)]
    # for name in bond_id_names:
    #     bond_feature_id = CompoundKit.get_bond_feature_size(name) + 2   # N + 2: self loop
    #     data[name] += [bond_feature_id] * N

    ### check whether edge exists
    if len(data['edges']) == 0: # mol has no bonds
        for name in bond_id_names:
            data[name] = np.zeros((0,), dtype="int64")
        data['edges'] = np.zeros((0, 2), dtype="int64")

    ### make ndarray and check length
    for name in atom_id_names:
        data[name] = np.array(data[name], 'int64')
    data['mass'] = np.array(data['mass'], 'float32')
    for name in bond_id_names:
        data[name] = np.array(data[name], 'int64')
    data['edges'] = np.array(data['edges'], 'int64')

    ### morgan fingerprint
    data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), 'int64')
    # data['morgan2048_fp'] = np.array(CompoundKit.get_morgan2048_fingerprint(mol), 'int64')
    data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), 'int64')
    data['daylight_fg_counts'] = np.array(CompoundKit.get_daylight_functional_group_counts(mol), 'int64')
    return data


def mol_to_geognn_graph_data(mol, atom_poses, dir_type):
    """
    mol: rdkit molecule
    dir_type: direction type for bond_angle grpah
    """
    if len(mol.GetAtoms()) == 0:
        return None

    data = mol_to_graph_data(mol)

    data['atom_pos'] = np.array(atom_poses, 'float32')
    data['bond_length'] = Compound3DKit.get_bond_lengths(data['edges'], data['atom_pos'])
    BondAngleGraph_edges, bond_angles, bond_angle_dirs = \
            Compound3DKit.get_superedge_angles(data['edges'], data['atom_pos'])
    data['BondAngleGraph_edges'] = BondAngleGraph_edges
    data['bond_angle'] = np.array(bond_angles, 'float32')
    return data


def mol_to_geognn_graph_data_MMFF3d(mol):
    """tbd"""
    if len(mol.GetAtoms()) == 400:
        mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=10)
    else:
        atom_poses = Compound3DKit.get_2d_atom_poses(mol)
    return mol_to_geognn_graph_data(mol, atom_poses, dir_type='HT')


def mol_to_geognn_graph_data_raw3d(mol):
    """tbd"""
    atom_poses = Compound3DKit.get_atom_poses(mol, mol.GetConformer())
    return mol_to_geognn_graph_data(mol, atom_poses, dir_type='HT')



# def atom_features(atom):
#     return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
#                                           ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'Pd', 'Co',  'H',
#                                            'Pt', 'Unknown']) +
#                     one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
#                     one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
#                     one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
#                     [atom.GetIsAromatic()])
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
import networkx as nx
def get_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(bond_feats)


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():

        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index
# def smile_to_graph(smile):
#     mol = Chem.MolFromSmiles(smile)
#
#     c_size = mol.GetNumAtoms()
#
#     features = []
#     for atom in mol.GetAtoms():
#         feature = atom_features(atom)
#         features.append(feature / sum(feature))
#
#     edges = []
#     # edges_type=[]
#     for bond in mol.GetBonds():
#         bond_type = get_bond_features(bond)
#         bond_type = np.argmax(bond_type)
#         edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type])
#         # edges_type.append(bond_type)
#     g = nx.Graph()
#     g.add_weighted_edges_from(edges)
#     g = g.to_directed()
#     edgewidth = []
#     edge_index = []
#     for e1, e2 in g.edges:
#         edge_index.append([e1, e2])
#         a = list(g.get_edge_data(e1, e2).values())[0]
#         edgewidth.append(a)
#     return c_size, features, edge_index,edgewidth
import torch
from torch_geometric.utils import dense_to_sparse
def make_one_hot1(data1):
    a = (np.arange(100) == data1[:, None]).astype(np.integer)

    return a
def make_one_hot2(data1):
    a = (np.arange(25) == data1[:, None]).astype(np.integer)

    return a
def make_one_hot_atom(data,num):
    a = (np.arange(num) == data[:, None]).astype(np.integer)
    return a
def bond_to_edge(bond):
    src = bond.GetBeginAtomIdx()
    dst = bond.GetEndAtomIdx()
    bond_type = bond.GetBondTypeAsDouble()
    edge = [src, dst, bond_type]
    return edge
def sgat3d(mol):
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
    return W_spr
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
    for smile in compound_iso_smiles:
        mol = AllChem.MolFromSmiles(smile)
        nodes = []
        for atom in mol.GetAtoms():
            nodes.append(atom_to_node(atom))
        idx = [n[0] for n in nodes]
        nodes = np.array(nodes, dtype=float)[:, 1:]
        atom_num = torch.LongTensor(nodes[:, 0])
        atom_num_oh = torch.nn.functional.one_hot(atom_num, 79)
        node_feats = torch.FloatTensor(nodes[:, 1:])
        x = torch.cat((node_feats, atom_num_oh.to(torch.float)), dim=1)
        x=np.array(x)
        c_size = mol.GetNumAtoms()
        features = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            feature= atom_features3(atom)
            features.append(feature)

            print('d')
        #features=atom_features1(mol)
        sgat=sgat3d(mol)
        data = mol_to_geognn_graph_data_MMFF3d(mol)
        bond_f1=make_one_hot1(data['bond_type'])[:,[2,3,4,13,99]]
        #[:, [2, 3, 4, 13, 24]]
        #bond_f2=make_one_hot2(data['bond_dir'])
        #bond_f3 = make_one_hot1(data['is_in_ring'])[:,[1,2]]
        bond_f3 = np.array(data['is_in_ring']-1).reshape(-1,1)
        #bond_f1 = np.array(data['bond_type']-1).reshape(-1,1)
        #bond_f4 = make_one_hot1(data['bond_dir'])[:, [1,4,5]]
        z=edge(data['bond_type'])
        bond_f = np.vstack((z, data['is_in_ring'])).T
        #bond_f=np.hstack((bond_f1,bond_f3))
        #bond_f2 = make_one_hot2(data['bond_dir'])
        bond_type.extend((data['bond_type']))
        bond_dir.extend(data['bond_dir'])

        atom_Num.extend(data['atomic_num'])
        degree.extend((data['degree']))
        totol_hs.extend(data['total_numHs'])
        implicit_valence.extend(data['implicit_valence'])
        hybridization.extend(data['hybridization'])
        formal_charge.extend(data['formal_charge'])
        chiral_tag.extend(data['chiral_tag'])
        #GetNumRadicalElectrons.extend(data['chiral_tag'])
        atom1= make_one_hot1(data['atomic_num'])[:, [5,6,7,8,9,15,16,17,35,53,78,99]]
        atom2=make_one_hot1(data['degree'])[:, [1,2,3,4,5,99]]
        atom3 = make_one_hot1(data['total_numHs'])[:, [1, 2, 3,4,99]]
        atom4 = make_one_hot1(data['implicit_valence'])[:, [1, 2, 3, 4,99]]
        atom5 = make_one_hot1(data['is_aromatic'])[:, [1, 2]]
        atom6 = make_one_hot1(data['formal_charge'])[:, [5, 6, 7, 8,99]]
        atom7= make_one_hot1(data['hybridization'])[:, [3, 4, 5, 6,99]]
        atom8=make_one_hot1(data['chiral_tag'])[:, [1,2,3,99]]
        # atom=np.hstack((atom1,atom2,atom3,atom4,atom5,atom6,atom7,atom8))
        atom = np.hstack((atom1, atom2, atom3, atom4, atom5))
        features=np.vstack((features,data['degree'],data['total_numHs'],data['implicit_valence'],data['is_aromatic']))
        # aa=np.sum(atom,1)
        # atom=atom/aa
        features=features.T
        print('d')
        # z = np.where(~bond_f[:, :5].any(axis=1))[0]
        # bond_f[:, bond_f.shape[1] - 1][z] = 1
        # assert  bond_f.shape[1]==
        # l=np.max(data['bond_type'])
        # if l >max1 :
        #     max1=l
        #     print(max1)
        # z.append(type)
        #print(features.shape)
        smile_graph[smile] =  c_size,features,data['edges'],bond_f,data['bond_length'],sgat,data['BondAngleGraph_edges'],data['bond_angle'],data['atom_pos'],data['daylight_fg_counts']
        print('d')
    # z1=np.unique(atom_Num)
    # z2 = np.unique(degree)
    # z3 = np.unique(totol_hs)
    # z4 = np.unique(implicit_valence)
    # z5=np.unique(formal_charge)
    # z6 = np.unique(hybridization)
    # z7=np.unique(bond_type)
    # z8=np.unique(bond_dir)
    # z9=np.unique(chiral_tag)
    #z9 = np.unique(GetNumRadicalElectrons)
    np.save('../graph2d1.npy', smile_graph)
    #np.save('../graph3d1.npy', smile_graph)
    # for smile in compound_iso_smiles:
    #     g = smile_to_graph(smile)
    #     mol = AllChem.MolFromSmiles(smile)
    #     data = mol_to_geognn_graph_data_MMFF3d(mol)
    #     smile_graph[smile] = g
    # for smile in compound_iso_smiles:
    #     mol = AllChem.MolFromSmiles(smile)
    #     data = mol_to_geognn_graph_data_MMFF3d(mol)
    #     # g = smile_to_graph(smile)
    #     smile_graph_3d[smile] = data
    # print('dd')