import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.conv import GINConv

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, sparse):
        super(GIN, self).__init__()

        if not sparse:
            raise NotImplementedError
        else:
            self.gnn = GIN_sparse(in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj)

    def forward(self, x, adj=None, g=None):
        return self.gnn.forward(x, adj, g)


class GIN_sparse(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj):
        super(GIN_sparse, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(GINConv(nn.Linear(in_channels, hidden_channels), 'sum'))
        for _ in range(num_layers - 2):
            self.layers.append(GINConv(nn.Linear(hidden_channels, hidden_channels), 'max'))
        self.layers.append(GINConv(nn.Linear(hidden_channels, out_channels), 'max'))

        self.dropout = dropout
        self.dropout_adj_p = dropout_adj

    def forward(self, x, adj=None, g=None):

        g.edata['w'] = F.dropout(g.edata['w'], p=self.dropout_adj_p, training=self.training)
        
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(g, x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](g, x)
        return x




class GINConv_dgl(nn.Module):
    def __init__(self, input_size, output_size):
        super(GINConv_dgl, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        )
        self.eps = nn.Parameter(torch.zeros(1))  # Learnable parameter ε

    def forward(self, x, g):
        with g.local_scope():
            g.ndata['h'] = x
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_neigh'))
            x = (1 + self.eps) * x + g.ndata['h_neigh']  # Aggregation with learnable ε
            return self.mlp(x)  # Apply MLP