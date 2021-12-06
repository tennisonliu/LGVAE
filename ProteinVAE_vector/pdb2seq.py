from proteingraph import read_pdb
import numpy as np


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
protein2letter_dict = {"ALA":"A", "ARG":"R", "ASN":"N", "ASP":"D", "CYS":"C", "GLN":"Q", "GLU":"E", "GLY":"G", "HIS":"H", "ILE":"I", "LEU":"L", "LYS":"K", "MET":"M", "PHE":"F", "PRO":"P", "SER":"S", "THR":"T", "TRP":"W", "TYR":"Y", "VAL":"V", "SEC":"U", "PYL":"O", "XAA":"X", "UNK":"X", "ASX":"B", "GLX":"Z", "XLE":"J"}

def construct_string(p):
    """ construct the string that represents the amino acid sequence """
    node_keys = list(p.nodes.keys()) # get node and edge info
    # construct sequence string
    strings = []
    for item in node_keys:
        code = p.nodes[item]['residue_name'] # p.nodes[item]: {'chain_id': 'B', 'residue_number': 98, 'residue_name': 'GLN', 'x_coord': 10.266, 'y_coord': 4.412, 'z_coord': -14.444, 'features': None}
        strings.append(protein2letter_dict[code])
    return strings
