import torch
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from torchvision.io import decode_image
import pandas as pd

class CIFAR10Dataset(Dataset):
    def __init__(self, root, train, transform):
        self.root = root
        self.train = train
        self.transform = transform
        self.num_train_samples = 45000
        self.num_test_samples = 5000
        self.classes = {
            "airplane": 0,
            "automobile": 1,
            "bird": 2,
            "cat": 3,
            "deer": 4,
            "dog": 5,
            "frog": 6,
            "horse": 7,
            "ship": 8,
            "truck": 9,
        }
        self.label_file = pd.read_csv(f"{root}/trainLabels.csv").to_numpy()

    def __len__(self):
        if self.train:
            return self.num_train_samples
        else:
            return self.num_test_samples

    def __getitem__(self, index):
        if self.train:
            img = decode_image(f"{self.root}/train/{index+1}.png").to(torch.float32)
            label = self.classes[self.label_file[index][1]]
        else:
            img = decode_image(f"{self.root}/test/{index+1+self.num_train_samples}.png").to(torch.float32)
            label = self.classes[self.label_file[index+self.num_train_samples][1]]

        return self.transform(img), label




transform = (transforms.Compose([
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
]))

dataset = CIFAR10Dataset("/home/sabyasachi19/PycharmProjects/CVBootcamp/cifar10_from_scratch/cifar-10",
                         train=True, transform=transform)

print(dataset[0])
#
# train_data = CIFAR10(root='./data', train=True, transform=transform, download=True)
# test_data = CIFAR10(root='./data', train=False, transform=transform, download=True)
#
# train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True, pin_memory=True)
# test_data_loader = DataLoader(test_data, batch_size=64, shuffle=False, pin_memory=True)
#
# train_batch = next(iter(train_data_loader))
# print(train_batch)

# print(sorted(os.listdir("/home/sabyasachi19/PycharmProjects/CVBootcamp/cifar10_from_scratch/cifar-10/train")))