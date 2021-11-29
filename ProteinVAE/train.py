from proteingraph import read_pdb
import numpy as np
import networkx as nx # tutorial: https://networkx.org/documentation/stable/tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import LGVAE, ELBO
from utils import construct_new_graph
from dataloader import get_dataloader, collate_pool, ProteinData
from hyperparams import is_padding, padding_num, is_CUDA

learning_rate = 1e-4

dataset = ProteinData(is_padding, padding_num)

train_loader, test_loader = get_dataloader(dataset,train_ratio=0.9, test_ratio=0.1, batch_size=2)

model = LGVAE(5, 2, num_of_nodes_in_protein=padding_num)
if is_CUDA==True:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for i, (x, A) in enumerate(train_loader):
    optimizer.zero_grad()
    x_out, A_out, mu, logsigma = model(x, A)
    # x_out shape: (batch_size, num_of_nodes_in_protein, feature_length)
    # A_out shape: (batch_size, num_of_nodes_in_protein, num_of_nodes_in_protein)
    a = x*torch.log(x_out)
    loss = ELBO(x, x_out, A, A_out, mu, logsigma)
    loss.backward()
    optimizer.step()
    print("Epoch[%d][%d/%d]     loss %.4f"  %(0, i, len(train_loader), loss.detach().cpu()))

print("Done")
