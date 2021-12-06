from proteingraph import read_pdb
import numpy as np
import networkx as nx # tutorial: https://networkx.org/documentation/stable/tutorial.html


# Download coordinate files in PDB Format:
# rsync -rlpt -v -z --delete --port=33444 \
# rsync.rcsb.org::ftp_data/structures/divided/pdb/ ./pdb
# For more info, see https://www.wwpdb.org/ftp/pdb-ftp-sites#rcsbpdb
# and https://ftp.rcsb.org/pub/pdb/data/structures/divided/pdb/

# # draw graph
# nx.draw(p)
# plt.savefig("path.png")
# plt.show()

protein2onehot_dict = {"ALA":0, "ARG":1, "ASN":2, "ASP":3, "CYS":4, "GLN":5, "GLU":6, "GLY":7, "HIS":8, "ILE":9, "LEU":10, "LYS":11, "MET":12, "PHE":13, "PRO":14, "SER":15, "THR":16, "TRP":17, "TYR":18, "VAL":19, "ASX": 20, "GLX": 20, "XLE":20, "UNK":20, "XAA":20}
protein2letter_dict = {"ALA":"A", "ARG":"R", "ASN":"N", "ASP":"D", "CYS":"C", "GLN":"Q", "GLU":"E", "GLY":"G", "HIS":"H", "ILE":"I", "LEU":"L", "LYS":"K", "MET":"M", "PHE":"F", "PRO":"P", "SER":"S", "THR":"T", "TRP":"W", "TYR":"Y", "VAL":"V"}

def protein2onehot(code):
    position = protein2onehot_dict[code]
    one_hot_vec = np.zeros(21)
    one_hot_vec[position] = 1.0
    return one_hot_vec

# construct feature matrix of p
# Maybe this is helpful: https://networkx.org/documentation/stable/reference/generated/networkx.linalg.attrmatrix.attr_matrix.html
def construct_new_graph(p, is_padding = False, padding=5000):
    node_keys = list(p.nodes.keys()) # get node and edge info
    edge_keys = list(p.edges.keys())
    # construct a new graph G w/ or without padding
    G = nx.Graph()
    i = 0
    for item in node_keys:
        code = p.nodes[item]['residue_name'] # p.nodes[item]: {'chain_id': 'B', 'residue_number': 98, 'residue_name': 'GLN', 'x_coord': 10.266, 'y_coord': 4.412, 'z_coord': -14.444, 'features': None}
        one_hot_vec = protein2onehot(code)
        G.add_nodes_from([(i, {"features": one_hot_vec})])
        i = i + 1
    # padding
    if is_padding == True:
        for i in range(p.number_of_nodes(), padding):
            G.add_nodes_from([(i, {"features": np.zeros(21)})])
    # construct edge info
    for item in edge_keys:
        G.add_edges_from([(node_keys.index(item[0]), node_keys.index(item[1]))])
    return G
