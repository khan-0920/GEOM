from utils.dataset import *
from model.GAT import *
from torch_geometric.data import DataLoader


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data) 
        loss = criterion(out, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y.view(-1, 1))
            total_loss += loss.item()
            
    return total_loss / len(loader)

def main():
    
    
    qm9_dataset = GeomDataset("../tmp")

    qm9_dataset.load("/gpfs/share/home/1800011712/GEOM/tmp/100000_total_data_qm9_withdis.pt")
    
    drugs_dataset = GeomDataset("../tmp")
    
    drugs_dataset.load("/gpfs/share/home/1800011712/GEOM/tmp/100000_total_data_drugs_withdis.pt")

    test_loader = DataLoader(qm9_dataset, batch_size=32)
    
    train_loader = DataLoader(drugs_dataset,batch_size=32, shuffle=True)
    
    device = torch.device("cuda:0")
    
    #model = geomGAT(176,176,179,1).to(device)
    model = geomGAdisT(176,176,176,1,14).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(1, 51):
        
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}')
        
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f'Epoch {epoch:03d}, Test Loss: {test_loss:.4f}')
        
if __name__ == "__main__":
    
    main()