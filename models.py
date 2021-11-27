"""
PUT ONLY PYTORCH MODULES/LAYERS/NETWORK DEFINITIONS IN HERE

TRAINING AND DATA PREPROCESSING HAPPENS IN DIFFERENT SCRIPTS
"""


from proteingraph import read_pdb
import numpy as np
import networkx as nx # tutorial: https://networkx.org/documentation/stable/tutorial.html
import torch
import torch.nn as nn
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GCNConv

from utils import construct_new_graph


class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(20, 32)
        self.conv2 = GCNConv(32, 20)
        self.activation = nn.ReLU()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index # now x has shape (num_nodes, 20)
        x = self.conv1(x, edge_index) # now x has shape (num_nodes, 32)
        x = self.activation(x)
        x = self.conv2(x, edge_index) # now x has shape (num_nodes, 20)
        return x

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseGCNConv as GCNConv
from torch_geometric.nn import dense_diff_pool

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))

        self.activations = torch.nn.ModuleList()
        self.activations.append(nn.ReLU())
        self.activations.append(nn.ReLU())

    def forward(self, x, adj, mask=None):
        for step in range(len(self.convs)):
            x = self.activations[step](self.convs[step](x, adj, mask))
        return x # graph with shape (num_nodes, out_channels)


class DiffPoolEn(nn.Module):
    def __init__(self):
        super(DiffPoolEn, self).__init__()

        num_nodes1 = 5
        self.gnn1_pool = GNN(20, 64, num_nodes1) # in_channels, hidden_channels, out_channels
        self.gnn1_embed = GNN(20, 64, 64)

        num_nodes2 = 2
        self.gnn2_pool = GNN(64, 64, num_nodes2)
        self.gnn2_embed = GNN(64, 64, 64, lin=False)

        self.gnn3_embed = GNN(64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(64, 64)
        self.lin2 = torch.nn.Linear(64, 2)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask) # torch.Size([1, num_nodes, 5])
        x = self.gnn1_embed(x, adj, mask) # torch.Size([1, num_nodes, 64])
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask) # see https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html
        # now x has shape torch.Size([1, 5, 64])
        # adj has shape torch.Size([1, 5, 5])

        s = self.gnn2_pool(x, adj) # torch.Size([1, 5, 2])
        x = self.gnn2_embed(x, adj) # torch.Size([1, 5, 64])
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)
        # now x has shape torch.Size([1, 2, 64])
        # adj has shape torch.Size([1, 2, 2])

        x = self.gnn3_embed(x, adj) # torch.Size([1, 2, 64])


        """CONSTRUCT MU AND SIGMA FUNCTION HERE!!!"""

        x = x.mean(dim=1) # torch.Size([1, 64])
        x = F.relu(self.lin1(x)) # torch.Size([1, 64])
        x = self.lin2(x) # torch.Size([1, 2])
        out = F.log_softmax(x, dim=-1) # torch.Size([1, 2])
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2

class DiffPoolDec(nn.Module):
    def __init__(self):
        super(DiffPoolDec, self).__init__()

        num_nodes1 = 2
        self.gnn1_pool = GNN(64, 64, num_nodes1) # in_channels, hidden_channels, out_channels
        self.gnn1_embed = GNN(64, 64, 64)

        num_nodes2 = 5
        self.gnn2_pool = GNN(64, 64, num_nodes2)
        self.gnn2_embed = GNN(64, 64, 64, lin=False)

        num_nodes3 = 500
        self.gnn3_pool = GNN(64, 64, num_nodes3)
        self.gnn3_embed = GNN(64, 64, 20, lin=False)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask) # torch.Size([1, 1, 2])
        x = self.gnn1_embed(x, adj, mask) # torch.Size([1, 1, 64])
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask) # see https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html
        # now x has shape torch.Size([1, 2, 64])
        # adj has shape torch.Size([1, 2, 2])

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)


        s = self.gnn3_pool(x, adj)
        x = self.gnn3_embed(x, adj)
        x, adj, l3, e3 = dense_diff_pool(x, adj, s)
        
        return x, adj, l1 + l2 + l3, e1 + e2 + e3


class LGVAE(nn.Module):
    def __init__(self):
        super(LGVAE, self).__init__()
        self.encoder = DiffPoolEn()
        self.decoder = DiffPoolDec()
    def forward(self, x, adj, mask=None):
        # encode
        mu_logsigma, _ , _ = self.encoder(x, adj, mask=None)
        # sampling through mu and log_sigma
        mu = torch.exp(mu_logsigma[0,0]).detach().numpy().tolist()
        sigma = torch.exp(mu_logsigma[0,1]).detach().numpy().tolist()
        z_0 = torch.normal(mu, sigma, size=(1, 1, 64))
        z = sigma*z_0 + mu
        adj = torch.from_numpy(np.array([1.0]).astype("float32")).view(1,1,1)
        # decode
        x_out, adj_out, _, _ = self.decoder(z, adj, mask=None)
        print(x_out.shape, adj_out.shape)


p = read_pdb('test.pdb')

q = construct_new_graph(p, is_padding=True)

A = nx.adjacency_matrix(q).todense().astype("float32") # numpy 2D matrix of shape (len(q), len(q))

A = torch.from_numpy(A)

# turn to a torch_geometric.data type
G = from_networkx(q)
G.x = G.features.float()
G.num_nodes = None
G.features = None

model = LGVAE()

model(G.x, A, mask=None)
