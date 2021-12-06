# load protein structure and convert to tensors
import functools
import networkx as nx # tutorial: https://networkx.org/documentation/stable/tutorial.html
import numpy as np
import os
from proteingraph import read_pdb
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.utils.convert import from_networkx

from utils import construct_new_graph
from hyperparams import is_CUDA


# load the RCSB dataset (sub-dataset which was cleaned in advance)
class ProteinData(Dataset):
    """
    Load the RCSB dataset
    """
    def __init__(self, is_padding, padding_num, root_dir= "/home/mingzeya/rcsb_protein/subset_database_cleaned_2000/", random_seed=123):
        self.root_dir = root_dir
        self.is_padding = is_padding # True or False
        self.padding_num = padding_num
        # get file list and shuffle
        self.all_files = os.listdir(root_dir)
        random.seed(random_seed)
        random.shuffle(self.all_files)

    def __len__(self):
        return len(self.all_files)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        target = self.all_files[idx]
        p = read_pdb(os.path.join(self.root_dir, target))
        q = construct_new_graph(p, is_padding=self.is_padding, padding = self.padding_num)
        A = nx.adjacency_matrix(q).todense().astype("float32") # numpy 2D matrix of shape (len(q), len(q))
        A = torch.from_numpy(A).view(1, A.shape[0], A.shape[1])
        # turn to a torch_geometric.data type
        G = from_networkx(q)
        G.x = G.features.float()
        G.num_nodes = None
        G.features = None
        return G, A # return the graph G and the adjacency matrix A

global padding_num_global
from hyperparams import padding_num as padding_num_global
# collate func
def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch
    dataset_list: list of tuples for each data point, i.e. [(G1, A1), (G2, A2), ...]
    """
    # get size of G.x and A to construct the graph
    x = torch.Tensor()
    x = torch.zeros(0, padding_num_global, 21)
    A = torch.zeros(0, padding_num_global, padding_num_global)
    for i, (Gi, Ai) in enumerate(dataset_list):
        x = torch.cat((x, Gi.x.unsqueeze(0)), dim=0)
        A = torch.cat((A, Ai), dim=0)
    if is_CUDA == True:
        x = x.cuda()
        A = A.cuda()
    return x, A

def get_dataloader(dataset, collate_fn=collate_pool,
                              batch_size=8, train_ratio=0.9, test_ratio=0.1):
    total_size = len(dataset)
    assert train_ratio+test_ratio <= 1.0

    indices = list(range(total_size))
    train_size = int(train_ratio * total_size)
    test_size = int(test_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=collate_fn)
    test_loader = DataLoader(dataset, batch_size=batch_size,
                             sampler=test_sampler,
                             collate_fn=collate_fn)

    return train_loader, test_loader
