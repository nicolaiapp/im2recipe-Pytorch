import os
import random
import numpy as np
import pickle
import sys

sys.path.append("..")
from model.args import get_parser

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

random.seed(opts.seed)
type_embedding = opts.embtype
# 'image'
# type_embedding = 'recipe'
with open(os.path.join(opts.path_results, 'img_embeds.pkl'), 'rb') as f:
    im_vecs = pickle.load(f)
with open(os.path.join(opts.path_results, 'rec_embeds.pkl'), 'rb') as f:
    instr_vecs = pickle.load(f)
with open(os.path.join(opts.path_results, 'rec_ids.pkl'), 'rb') as f:
    names = pickle.load(f)
with open(os.path.join(opts.path_results, 'img_ids.pkl'), 'rb') as f:
    img_names = pickle.load(f)


with open(os.path.join(opts.path_results, 'test_keys.pkl'), 'rb') as f:
    test_keys = pickle.load(f)


# Sort based on names to always pick same samples for medr
idxs = np.argsort(names)
names = names[idxs]

# for id in names:
#     print('id: ' + id)
#
# recipe_instr_vec = instr_vecs[0, :]
# print(recipe_instr_vec)
#
# for instr_vec in instr_vecs:
#     print('instr_vec')
#     print(instr_vec)
#
# for name in names:
#     print('name')
#     print(name)
#
# for name in img_names:
#     print('img_name')
#     print(name)


print('recipe 00003a70b1 has index: ', test_keys.index('00003a70b1'))
print('recipe f6af7320c4 has index: ', test_keys.index('f6af7320c4'))


