from __future__ import absolute_import, print_function
import os
from DataSet import transforms
import torchvision.datasets as datasets


class Car196:
    def __init__(self, root, train=True, test=True, transform=None):
        # Data loading code
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]

        if transform is None:
            transform = [transforms.Compose([
                #transforms.CovertBGR(),
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_values,
                                     std=std_values),
            ]),
                transforms.Compose([
                    #transforms.CovertBGR(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean_values,
                                         std=std_values),
                ])]

        if root is None:
            root = 'DataSet/Car196'
        traindir = os.path.join(root, 'train')
        testdir = os.path.join(root, 'test')

        if train:
            self.train = datasets.ImageFolder(traindir, transform[0])
        if test:
            self.test = datasets.ImageFolder(testdir, transform[1])

