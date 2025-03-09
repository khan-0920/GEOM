#import torch_geometric
from utils.dataset import *
import rdkit
import os
import json
from torch_geometric.data import Dataset

def generate_database():
    
    dir_ = "/gpfs/share/home/1800011712/GEOM/data/rdkit_folder/"
    
    with open(dir_+"summary_qm9.json","r") as f:
        data_dict = json.load(f)
    
    counter = 0
    up_limit = 100000
    
    keys = data_dict.keys()
    dataset = GeomDataset("tmp")
    
    for key in keys:
        
        if counter > up_limit:
            break
        
        else:
            try:
                with open(os.path.join(dir_,"qm9",str(key+".pickle")),"rb") as f: data = pickle.load(f)
                if len(data["conformers"]) > 2:
                    for i in range(len(data["conformers"])):
                        dataset.add(data,i)
                        counter += 1
            except:
                print(key)
                                    
    dataset.save(f"tmp/{up_limit}_total_data_qm9.pt")

def load_database():
    
    dataset = GeomDataset("tmp")
    dataset.load("100_total_data_qm9.pt")
    print(dataset.len())
  
if __name__ == "__main__":
    
    generate_database()