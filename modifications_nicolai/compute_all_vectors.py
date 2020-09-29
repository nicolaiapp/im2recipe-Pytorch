import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
from data_loader import ImageLoader  # our data_loader
import numpy as np
from trijoint import im2recipe
# from trijoint_ingredient import im2recipe
import pickle
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
# np.set_printoptions(suppress=True)

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
    # model.visionMLP = torch.nn.DataParallel(model.visionMLP, device_ids=[0, 1, 2, 3])
    # model.visionMLP = torch.nn.DataParallel(model.visionMLP, device_ids=[0,1])
    model.visionMLP = torch.nn.DataParallel(model.visionMLP, device_ids=[0])
    if not opts.no_cuda:
        model.cuda()

    # define loss function (criterion) and optimizer
    # cosine similarity between embeddings -> input1, input2, target
    cosine_crit = nn.CosineEmbeddingLoss(0.1)
    if not opts.no_cuda:
        cosine_crit.cuda()
    # cosine_crit = nn.CosineEmbeddingLoss(0.1)
    if opts.semantic_reg:
        weights_class = torch.Tensor(opts.numClasses).fill_(1)
        weights_class[0] = 0  # the background class is set to 0, i.e. ignore
        # CrossEntropyLoss combines LogSoftMax and NLLLoss in one single class
        class_crit = nn.CrossEntropyLoss(weight=weights_class)
        if not opts.no_cuda:
            class_crit.cuda()
        # class_crit = nn.CrossEntropyLoss(weight=weights_class)
        # we will use two different criteria
        criterion = [cosine_crit, class_crit]
    else:
        criterion = cosine_crit

    print("=> loading checkpoint '{}'".format(opts.model_path))
    checkpoint = torch.load(opts.model_path)
    opts.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(opts.model_path, checkpoint['epoch']))

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # preparing test loader 
    test_loader = torch.utils.data.DataLoader(
        ImageLoader(opts.img_path,
                    transforms.Compose([
                        transforms.Resize(256),  # rescale the image keeping the original aspect ratio
                        transforms.CenterCrop(224),  # we get only the center of that rescaled
                        transforms.ToTensor(),
                        normalize,
                    ]), data_path=opts.data_path, sem_reg=opts.semantic_reg, partition='test'),
        batch_size=opts.batch_size, shuffle=False,
        num_workers=opts.workers, pin_memory=(not opts.no_cuda))
    print('Test loader prepared.')

    # run test
    test(test_loader, model, criterion)


def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    cos_losses = AverageMeter()
    if opts.semantic_reg:
        img_losses = AverageMeter()
        rec_losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    limit = 1
    # recipe_id = 'f6af7320c4'  has index 49568 in test_keys.pkl
    # for i, (input_, target) in izip(range(limit), test_loader):
    for i, (input_, target) in enumerate(test_loader):
        print('index i: ' + str(i))
        if i == limit:
            break
        # for i, (input_, target) in enumerate(list(test_loader)[0]):
        input_var, target_var = list(), list()
        # input: [img, instrs, itr_ln, ingrs, igr_ln]
        # target: [target, img_class, rec_class, img_id, rec_id]
        with torch.no_grad():
            for j in range(len(input_)):
                # print('len(input_): ', len(input_))
                # print('index j: ' + str(j))
                input_var.append(input_[j].cuda() if not opts.no_cuda else input_[j])
        with torch.no_grad():
            for j in range(len(target) - 2):  # we do not consider the last two objects of the list == [img_id, rec_id]
                # print('len(target): ', len(target))
                # print('index j: ' + str(j))
                target_var.append(target[j].cuda() if not opts.no_cuda else target[j])

        # compute output
        output = model(input_var[0], input_var[1], input_var[2], input_var[3], input_var[4])
        print(output[1].data.cpu().numpy())

        # # compute loss
        # if opts.semantic_reg:
        #     cos_loss = criterion[0](output[0].float(), output[1].float(), target_var[0].float())
        #     img_loss = criterion[1](output[2], target_var[1])
        #     rec_loss = criterion[1](output[3], target_var[2])
        #     # combined loss
        #     loss = opts.cos_weight * cos_loss + \
        #            opts.cls_weight * img_loss + \
        #            opts.cls_weight * rec_loss
        #
        #     # measure performance and record losses
        #     cos_losses.update(cos_loss.data[0], input_[0].size(0))
        #     img_losses.update(img_loss.data[0], input_[0].size(0))
        #     rec_losses.update(rec_loss.data[0], input_[0].size(0))
        # else:
        #     loss = criterion(output[0], output[1], target_var[0])
        #     # measure performance and record loss
        #     cos_losses.update(loss.data[0], input_[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i == 0:
            data0 = output[0].data.cpu().numpy()
            data1 = output[1].data.cpu().numpy()
            # print('data1: ', str(data1))
            # print('data1.size: ', str(torch.from_numpy(data1).size()))
            data2 = target[-2]
            data3 = target[-1]
        else:
            data0 = np.concatenate((data0, output[0].data.cpu().numpy()), axis=0)
            data1 = np.concatenate((data1, output[1].data.cpu().numpy()), axis=0)
            data2 = np.concatenate((data2, target[-2]), axis=0)
            data3 = np.concatenate((data3, target[-1]), axis=0)

    if opts.semantic_reg:
        print('* Test cosine loss {losses.avg:.4f}'.format(losses=cos_losses))
        print('* Test img class loss {losses.avg:.4f}'.format(losses=img_losses))
        print('* Test rec class loss {losses.avg:.4f}'.format(losses=rec_losses))
    else:
        print('* Test loss {losses.avg:.4f}'.format(losses=cos_losses))

    with open(opts.path_results + 'img_embeds.pkl', 'wb') as f:
        pickle.dump(data0, f)
    with open(opts.path_results + 'rec_embeds.pkl', 'wb') as f:
        pickle.dump(data1, f)
    with open(opts.path_results + 'img_ids.pkl', 'wb') as f:
        pickle.dump(data2, f)
    with open(opts.path_results + 'rec_ids.pkl', 'wb') as f:
        pickle.dump(data3, f)

    return cos_losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
