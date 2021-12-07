# load protein structure and convert to tensors
import functools
import networkx as nx # tutorial: https://networkx.org/documentation/stable/tutorial.html
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.utils.convert import from_networkx

from hyperparams import is_CUDA


# load the RCSB dataset (sub-dataset which was cleaned in advance)
class ProteinData(Dataset):
    """
    Load the RCSB dataset by reading pkl file
    """
    def __init__(self, root_dir= '/home/mingzeya/rcsb_protein/pdb_tensor', random_seed=123):
        self.root_dir = root_dir
        # get file list and shuffle
        self.all_files = os.listdir(root_dir)
        random.seed(random_seed)
        random.shuffle(self.all_files)

    def __len__(self):
        return len(self.all_files)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        target = self.all_files[idx] # e.g., 1m1b.pkl
        with open(os.path.join(self.root_dir, target),"rb") as f:
            data=pickle.load(f)
        return data.detach().cpu().view(-1)


# collate func
def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch
    dataset_list: list of tuples for each data point, i.e. [vec1, vec2, ...]
    """
    # get size of G.x and A to construct the graph
    x = torch.zeros(0, 768)
    for i, vec in enumerate(dataset_list):
        x = torch.cat((x, vec.unsqueeze(0)), dim=0)
    if is_CUDA == True:
        x = x.cuda()
    return x

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