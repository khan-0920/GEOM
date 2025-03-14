import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch.nn import ReLU, Linear, Sequential, Dropout
from torch_geometric.nn import global_mean_pool


class geomGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels,hidden_dim,out_dim,layer_num=5,heads=8,):
        super(geomGAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=heads)  # 8个头
        self.conv2 = GATConv(8 * heads, out_channels, heads=1)  # 输出只有一个头

        self.posGAT1 = GATConv(3, 8, heads=heads)
        self.posGAT2 = GATConv(8 * heads, 3, heads=1)
        
        self.mlp = Sequential(
            Linear(hidden_dim, hidden_dim//2),
            ReLU(),
            Linear(hidden_dim//2, out_dim)
        )
        
        self.layer_num = layer_num
        
    def forward(self, data):
        
        #device = torch.device("cuda:0")
        
        x, edge_index, pos, batch,edge_attr = data.x, data.edge_index, data.pos, data.batch, data.edge_attr
             
        for i in range(self.layer_num):
            pos = F.relu(self.posGAT1(pos,edge_index,edge_attr=edge_attr))
            pos = self.posGAT2(pos,edge_index,edge_attr=edge_attr)
            x = F.relu(self.conv1(x, edge_index,edge_attr=edge_attr))  # 激活函�?            
            x = self.conv2(x, edge_index,edge_attr=edge_attr)  # 第二�?      
              
        pos = global_mean_pool(pos, batch)
        x = global_mean_pool(x, batch)
        
        Layer_1 = torch.cat([x,pos],dim=1)
        
        out = self.mlp(Layer_1)
        
        return out

class geomGAdisT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, out_dim, edge_dim, layer_num=5, heads=8, dropout_rate=0.3):
        super(geomGAdisT, self).__init__()
        
        self.dropout_rate = dropout_rate
        
        self.conv1 = GATConv(in_channels, 8, heads=heads, dropout=self.dropout_rate)  # 8个头
        self.conv2 = GATConv(8 * heads, out_channels, heads=1, dropout=self.dropout_rate)  # 输出只有一个头
        
        self.mlp = Sequential(
            Linear(hidden_dim, hidden_dim//2),
            ReLU(),
            Linear(hidden_dim//2, out_dim)
        )
        
        self.edge_mlp = Sequential(
            Linear(edge_dim+in_channels*2,edge_dim+in_channels),
            ReLU(),
            Linear(edge_dim+in_channels, edge_dim)
        )
        
        self.dropout = Dropout(self.dropout_rate)
        
        self.layer_num = layer_num    
    
    def forward(self, data):
        
        #device = torch.device("cuda:0")
        
        x, edge_index, pos, batch, edge_attr, dis, dis_index = data.x, data.edge_index, data.pos, data.batch, data.edge_attr, data.dis, data.dis_index
             
        for i in range(self.layer_num):
            
            x_tmp = F.relu(self.conv1(x,dis_index,edge_attr=dis))
            x_tmp = self.conv2(x_tmp,dis_index,edge_attr=dis)
            x_tmp = F.relu(self.conv1(x_tmp, edge_index,edge_attr=edge_attr))  # 激活函�?            
            x_tmp = self.conv2(x_tmp, edge_index,edge_attr=edge_attr)  # 第二�?      
            
            row, col = edge_index
            
            edge_input = torch.cat([x[row],x[col],edge_attr],dim=1)
            edge_attr = self.edge_mlp(edge_input)
            
            x = x_tmp
            
            x = self.dropout(x)
        
        x = global_mean_pool(x, batch)
        
        out = self.mlp(x)
        
        return out
