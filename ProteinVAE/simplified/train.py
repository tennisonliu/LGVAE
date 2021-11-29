from proteingraph import read_pdb
import numpy as np
import networkx as nx # tutorial: https://networkx.org/documentation/stable/tutorial.html
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import LGVAE, ELBO
from utils import construct_new_graph
from dataloader import get_dataloader, collate_pool, ProteinData
from hyperparams import is_padding, padding_num, is_CUDA

learning_rate = 1e-4

dataset = ProteinData(is_padding, padding_num, root_dir= "/home/mingzeya/rcsb_protein/subset_database_cleaned_2000/")

train_loader, test_loader = get_dataloader(dataset,train_ratio=0.9, test_ratio=0.1, batch_size=8)

model = LGVAE(5, num_of_nodes_in_protein=padding_num)
if is_CUDA==True:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Get working directory
working_dir = os.getcwd()
# Initialize synchronized output
fout = open("syn_output.txt",'w')
fout.write("")
fout.close()

# training
for epoch in range(0,100):
    for i, (x, A) in enumerate(train_loader):
        optimizer.zero_grad()
        x_out, A_out, mu, logsigma = model(x, A)
        # x_out shape: (batch_size, num_of_nodes_in_protein, feature_length)
        # A_out shape: (batch_size, num_of_nodes_in_protein, num_of_nodes_in_protein)
        a = x*torch.log(x_out)
        loss = ELBO(x, x_out, A, A_out, mu, logsigma)
        loss.backward()
        optimizer.step()
        output = "Epoch[%d][%d/%d]     loss %.4f"  %(epoch, i, len(train_loader), loss.detach().cpu())
        print(output)
        fout = open("syn_output.txt",'a')
        fout.write(output)
        fout.write("\n")
        fout.close()
    # save model each epoch
    model_name = 'model_epoch_'+str(epoch)+".pth"
    torch.save({'state_dict': model.state_dict()}, model_name)
    torch.cuda.empty_cache()

optimizer.zero_grad()


# test
for i, (x, A) in enumerate(test_loader):
    x_out, A_out, mu, logsigma = model(x, A)
    # x_out shape: (batch_size, num_of_nodes_in_protein, feature_length)
    # A_out shape: (batch_size, num_of_nodes_in_protein, num_of_nodes_in_protein)
    a = x*torch.log(x_out)
    loss = ELBO(x, x_out, A, A_out, mu, logsigma)
    output = "Test [%d/%d]     loss %.4f"  %(i, len(test_loader), loss.detach().cpu())
    print(output)
    fout = open("syn_output.txt",'a')
    fout.write(output)
    fout.write("\n")
    fout.close()

print("Done")
