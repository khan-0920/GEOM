import pickle
import os



from rdkit import Chem

def rd_to_data():
    
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