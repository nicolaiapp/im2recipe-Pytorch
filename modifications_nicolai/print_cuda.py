import torch

if torch.cuda.is_available():
    print('available')


print(torch.cuda.device_count())