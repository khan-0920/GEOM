import pickle
import os
import torch
from torch_geometric.data import Data 

from rdkit import Chem

allowable_features = {
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
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
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


def rd_to_data(molecule_dataset,conf_id,):
    """
    data
    x: [atom_types,...]
    edge_index: [2, num_edges]
    pos: [num_nodes, 3]
    edge_attr: [num_edges, features]
    y: boltzman distribution
    
    """
    mol = data['conformers'][1]['rd_mol']
    
    x = torch.tensor([atom.GetIdx() for atom in mol.GetAtoms()])
    
    pos = torch.tensor(mol.GetConformer().GetPositions())
    
    row, col, edge_type = [], [], []
    
    for bond in mol.GetBonds():
        
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += [bond.GetBondType()]
    
    edge_index = torch.tensor(row+col,dtype=torch.long)
    
    
    
    return NotImplementedError


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