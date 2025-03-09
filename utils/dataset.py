import pickle
import os
import torch
import numpy as np
from torch_geometric.data import Dataset, Data

from rdkit import Chem

class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            #s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def possible_atomic_num_list(self, atom):
        return atom.GetAtomicNum()

    def possible_chirality_list(self, atom):
        return atom.GetChiralTag()

    def possible_degree_list(self, atom):
        return atom.GetTotalDegree()

    def possible_formal_charge_list(self, atom):
        return atom.GetFormalCharge()
    
    def possible_numH_list(self, atom):
        return atom.GetTotalNumHs()
    
    def possible_number_radical_e_list(self, atom):
        return atom.GetNumRadicalElectrons()
    
    def possible_hybridization_list(self, atom):
        return atom.GetHybridization().name.lower()
    
    def possible_is_aromatic_list(self, atom):
        return atom.GetIsAromatic()
    
    def possible_is_in_ring_list(self, atom):
        return atom.IsInRing()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def possible_bond_type_list(self, bond):
        return str(bond.GetBondType())

    def possible_bond_stereo_list(self, bond):
        return str(bond.GetStereo())
    
    def possible_is_conjugated_list(self, bond):
        return bond.GetIsConjugated()


atom_featurizer = AtomFeaturizer(
    allowable_sets={
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
        "CHI_SQUAREPLANAR",
        "CHI_OTHER",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["sp", "sp2", "sp3", "sp3d", "sp3d2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True]
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
        "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
    }
)

def rd_to_data(conf_id,data):
    """
    
    conf_id: the conformation id of the data
    
    data
    x: [atom_feature,...]
    edge_index: [2, num_edges]
    pos: [num_nodes, 3]
    edge_attr: [num_edges, features]
    y: boltzman distribution
    
    """
    mol = data['conformers'][conf_id]['rd_mol']
    
    
    
    x = torch.tensor(np.array([atom_featurizer.encode(atom) for atom in mol.GetAtoms()]))
    
    pos = torch.tensor(mol.GetConformer().GetPositions())
    
    edge, edge_attr = [], []
    
    for bond in mol.GetBonds():
        
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge += [start,end]
        edge += [end,start]
        edge_attr += [bond_featurizer.encode(bond)]
        edge_attr += [bond_featurizer.encode(bond)]
    
    edge_index = torch.tensor(edge,dtype=torch.long)
    
    y = data['conformers'][conf_id]['boltzmannweight']
    
    output = Data(x=x, 
                  edge_index=edge_index, 
                  edge_attr=edge_attr, 
                  pos=pos, 
                  )
    
    return output

class GeomDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GeomDataset,self).__init__(root, transform, pre_transform)
        self.data_lst = []
        
    def len(self):
        return len(self.data_lst)
    
    def get(self,idx):
        return self.data_lst[idx]
    
    def load(self,pt_path):
        loaded_dataset = torch.load(pt_path)
        self.data_lst += loaded_dataset
    
    def save(self,out_path):
        torch.save(self.data_lst,out_path)
        
    def add(self,rd_data,conf_id):
        self.data_lst.append(rd_to_data(conf_id,rd_data))
        
if __name__ == "__main__":

    
    """
    direc = "/gpfs/share/home/1800011712/GEOM/data/"
    drugs_file = os.path.join(direc, "drugs_crude.msgpack")
    feature_file = os.path.join(direc, "drugs_crude.msgpack")
    unpacker = msgpack.Unpacker(open(drugs_file, "rb"))
    
    drugs_1k = next(iter(unpacker))
    
    sample_smiles = list(drugs_1k.keys())[10]
    sample_sub_dic = drugs_1k[sample_smiles]

    mol = Chem.MolFromSmiles(sample_smiles)
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        print(i)
        print(atom.GetSymbol())
    
    #print(Chem.MolToMolBlock(m))
    print(sample_sub_dic["conformers"][1]["xyz"])
    
    #print({key: val for key, val in sample_sub_dic.items() if key != 'conformers'})
    """
    
    file = ("/gpfs/share/home/1800011712/GEOM/data/rdkit_folder/drugs/BrC(_C=C\\c1ccccc1)=N_Nc1nc(N2CCOCC2)nc(N2CCOCC2)n1.pickle")
    
    with open(file,"rb") as f: data = pickle.load(f)
    
    print(data['conformers'][1]['rd_mol'])
    mol = data['conformers'][1]['rd_mol']
    print(data['conformers'][0]["boltzmannweight"])
    conf = mol.GetConformer()
    #print(conf.GetPositions())
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += [bond.GetBondType()]
    #print(row)
    #print(edge_type)

    
    """
    for atom in mol.GetAtoms():
        print(atom.GetSymbol())
    """