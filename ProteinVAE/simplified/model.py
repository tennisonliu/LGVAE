from proteingraph import read_pdb
import numpy as np
import networkx as nx # tutorial: https://networkx.org/documentation/stable/tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseGCNConv as GCNConv
from torch_geometric.nn import dense_diff_pool

from utils import construct_new_graph
from hyperparams import is_CUDA

# Graph NN layer
class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize)
        self.conv2 = GCNConv(hidden_channels, out_channels, normalize)
        self.activation = nn.ReLU()

    def forward(self, x, adj):
        x = self.activation(self.conv1(x, adj))
        x = self.conv2(x, adj)
        return x # shape (num_nodes, out_channels)

# Encoder, DiffPool
class DiffPoolEncoder(nn.Module):
    def __init__(self, num_nodes1):
        super(DiffPoolEncoder, self).__init__()

        self.gnn1_pool = GNN(21, 32, num_nodes1) # in_channels, hidden_channels, out_channels
        self.gnn1_embed = GNN(21, 32, 32)

        self.gnn3_embed = GNN(32, 32, 32)

        self.lin1_mu = torch.nn.Linear(32, 32)
        self.lin2_mu = torch.nn.Linear(32, 1)

        self.lin1_logsigma = torch.nn.Linear(32, 32)
        self.lin2_logsigma = torch.nn.Linear(32, 1)

    def forward(self, x, adj):
        s = self.gnn1_pool(x, adj) # torch.Size([1, num_nodes, 5])
        x = self.gnn1_embed(x, adj) # torch.Size([1, num_nodes, 32])
        x, adj, l1, e1 = dense_diff_pool(x, adj, s) # see https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html
        # now x has shape torch.Size([1, 5, 64])
        # adj has shape torch.Size([1, 5, 5])

        x = self.gnn3_embed(x, adj) # torch.Size([1, 2, 32])
        x = x.mean(dim=1) # torch.Size([1, 32])

        # mu func
        mu = self.lin2_mu(F.relu(self.lin1_mu(x))) # torch.Size([1, 1])
        logsigma = self.lin2_logsigma(F.relu(self.lin1_logsigma(x))) # torch.Size([1, 1])
        return mu, logsigma, l1  , e1


# Decoder, DiffPool
class DiffPoolDecoder(nn.Module):
    def __init__(self, num_nodes1,  output_nodes):
        super(DiffPoolDecoder, self).__init__()

        self.gnn1_pool = GNN(32, 32, num_nodes1) # in_channels, hidden_channels, out_channels
        self.gnn1_embed = GNN(32, 32, 32)

        self.gnn2_pool = GNN(32, 32, output_nodes)
        self.gnn2_embed = GNN(32, 32, 21)

        self.softmax_x = nn.Softmax(dim=2) # x shape: (batch_size, num_of_nodes_in_protein, feature_length)
        self.sigmoid_adj = nn.Sigmoid()

    def forward(self, x, adj):
        s = self.gnn1_pool(x, adj)
        x = self.gnn1_embed(x, adj)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s) # see https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.softmax_x(x)
        adj = self.sigmoid_adj(adj)

        return x, adj, l1 + l2 , e1 + e2


class LGVAE(nn.Module):
    def __init__(self, num_nodes1, num_of_nodes_in_protein = 1500):
        super(LGVAE, self).__init__()
        self.encoder = DiffPoolEncoder(num_nodes1)
        self.decoder = DiffPoolDecoder(num_nodes1, num_of_nodes_in_protein)
    def forward(self, x, adj):
        # encode
        batch_size = x.shape[0]
        mu, logsigma, _ , _ = self.encoder(x, adj)
        # sampling through mu and log_sigma
        if is_CUDA == True:
            mu_s = mu.cpu().view(-1).detach().numpy().tolist() # now mu is a list with len = 1
            sigma_s = torch.exp(logsigma).cpu().view(-1).detach().numpy().tolist()
        z = torch.zeros(0, 1, 32)
        for i in range(0,batch_size):
            mu_now = mu_s[i]
            sigma_now = sigma_s[i]
            z_0_now = torch.normal(mu_now, sigma_now, size=(1, 1, 32)) # batch_size, num_of_nodes, num of features
            z = torch.cat((z,sigma_now*z_0_now+mu_now), dim = 0)
        adj = torch.zeros(batch_size, 1, 1) + 1.0
        if is_CUDA == True:
            z = z.cuda()
            adj = adj.cuda()
        # decode
        x_out, adj_out, _, _ = self.decoder(z, adj)
        return x_out, adj_out, mu, logsigma


# ELBO loss
def ELBO(x, x_out, adj, adj_out, mu, logsigma):
    """ ELBO = -(x*log(x_out)+(1-x)*log(1-x_out)) -(adj*log(adj_out)+(1-adj)*log(1-adj_out)) - 1/2 (1 + logsigma - mu^2 - sigma) """
    ELBO_CE = -torch.sum(x*torch.log(x_out) + (1.0-x)*torch.log(1.0-x_out))/(x_out.shape[0]*x_out.shape[1]*x_out.shape[2]) -torch.sum(adj*torch.log(adj_out)+(1.0-adj)*torch.log(1.0-adj_out))/(adj.shape[0]*adj.shape[1]*adj.shape[2])
    batch_size = x.shape[0]
    ELBO_KL = - 0.5*(1.0*batch_size + torch.sum(logsigma) - torch.sum(mu**2) - torch.sum(torch.exp(logsigma)))/batch_size
    return ELBO_CE + ELBO_KL
