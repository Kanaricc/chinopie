import sys
sys.path.append('..')

import pdb
from chinopie.datasets import CIFAR100
from torchvision import transforms

if __name__=='__main__':
    a=CIFAR100('temp',train=True,preprocess=transforms.Compose([
        transforms.ToTensor()
    ]))
    print(a[0]['image'].size())