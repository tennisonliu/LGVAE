import pandas as pd
from pysmiles import read_smiles
import networkx as nx
import numpy as np

def get_atom_counts(data):
    '''
    Code to count the number of unique atoms that exist in a dataset
    '''
    unique_atoms_count = {}

    for row in data.itertuples():
        smiles = row.smiles
        mol = read_smiles(smiles)
        nodes = dict(mol.nodes(data='element'))
        unique_atoms_ = np.unique(list(nodes.values()))
        for atom in unique_atoms_:
            if atom not in unique_atoms_count:
                unique_atoms_count[atom] = 1
            else:
                unique_atoms_count[atom] += 1

    return unique_atoms_count


def process_smiles(smiles_string, atom_index):
    mol = read_smiles(smiles_string)
    nodes = mol.nodes(data='element')

    node_list = list(dict(nodes).values())
    atom_list = list(atom_index.keys())
    
    # if nodes are not in atom_list, then skip
    if len(np.setdiff1d(node_list, atom_list)) != 0:
        return

    num_nodes = mol.number_of_nodes()
    
    # one-hot encoded feature matrix
    feat_matrix = np.zeros((num_nodes, len(atom_index)))
    
    # put 1 (i.e. one-hot encoding based on atom type)
    for atom in nodes:
        feat_matrix[atom[0], atom_index[atom[1]]] = 1
        
    # adjacency matrix
    adj_matrix = nx.to_numpy_matrix(mol)
    
    # add identity matrix for self-loops
    adj_matrix += np.eye(adj_matrix.shape[0])
    
    return feat_matrix, adj_matrix

def preprocess_chembl(datapoints=10000):

    data = pd.read_csv('molecule-datasets/chembl_filtered_torchdrug.csv', delimiter=',', nrows=datapoints, usecols=['smiles'], index_col=False)
    atom_index = {
        "C": 0, 
        "O": 1, 
        "N": 2, 
        "Cl": 3, 
        "S": 4, 
        "F": 5, 
        "Br": 6, 
        "I": 7, 
        "P": 8
        }
    
    for row in data.itertupes():
        smiles = row.smiles
        res = process_smiles(smiles, atom_index)
        if res is not None:
            feat_mat, adj_mat = res
        


if __name__ == "__main__":
    main()