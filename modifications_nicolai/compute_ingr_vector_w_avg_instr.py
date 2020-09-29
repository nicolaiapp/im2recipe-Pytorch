import json
import pickle
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
from collections import OrderedDict
from trijoint_ingredient_avg_instr import im2recipe
from args import get_parser
import subprocess


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


np.set_printoptions(threshold=np.nan)

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

torch.manual_seed(opts.seed)
if not opts.no_cuda:
    torch.cuda.manual_seed(opts.seed)

np.random.seed(opts.seed)


def main():
    model = im2recipe()
    model.visionMLP = torch.nn.DataParallel(model.visionMLP, device_ids=[0])
    if not opts.no_cuda:
        model.cuda()

    print("=> loading checkpoint '{}'".format('pretrained_models/model_e220_v-4.700.pth.tar'))
    checkpoint = torch.load('pretrained_models/model_e220_v-4.700.pth.tar')
    opts.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(opts.model_path, checkpoint['epoch']))

    # data preparation, loaders
    # unique_ingredients_vec_keys = myhelper.json_path_to_list('data/recipe1M/w2v_found_unique_ingredients_keys.json')

    with open('data/recipe1M/avg_instr_vector.pkl', 'rb') as f:
        average_instr_vec = pickle.load(f)

    raw_json = open('data/recipe1M/w2v_w2index.json', mode='r')
    w2v_indices = json.load(raw_json, object_pairs_hook=OrderedDict)

    # w2v_indices = OrderedDict(reversed(list(w2v_indices.items())))

    recipe_list = list()
    # # ingrs = torch.tensor([[35]])
    igr_ln = torch.tensor([1])
    itr_ln = torch.tensor([1])
    instrs = torch.tensor([[average_instr_vec]])
    # recipe = [instrs, itr_ln, ingrs, igr_ln]
    # recipe = [input_var[0], input_var[1], input_var[2], input_var[3]]
    # recipe_list.append(recipe)
    for w2v_id in w2v_indices.values():
        if w2v_id >= 30165:  # ignore last two words for index reasons below
            continue
        recipe = [instrs, itr_ln, torch.tensor([[w2v_id + 2]]), igr_ln]  # +2 because lua skip-th and </s> token
        recipe_list.append(recipe)

    get_vectors(recipe_list, model)


def get_vectors(recipe_list, model):
    # switch to evaluate mode
    model.eval()

    for i, recipe in enumerate(recipe_list):
        input_var = list()
        with torch.no_grad():
            for j in range(len(recipe)):
                # print('len(input_): ', len(input_))
                # print('index j: ' + str(j))
                input_var.append(recipe[j].cuda() if not opts.no_cuda else recipe[j])

        # input: [img, instrs, itr_ln, ingrs, igr_ln]
        output = model(0, input_var[0], input_var[1], input_var[2], input_var[3])
        # print(output[1].data.cpu().numpy())

        if i == 0:
            data1 = output[1].data.cpu().numpy()
        else:
            data1 = np.concatenate((data1, output[1].data.cpu().numpy()), axis=0)

    with open(opts.path_results + 'ingredient_embeds_w_avg_instr_vec.pkl', 'wb') as f:
        pickle.dump(data1, f)


if __name__ == '__main__':
    main()
