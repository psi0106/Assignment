from torchvision.datasets import MNIST
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch


def get_transform():
    transform = v2.Compose([
        v2.ToTensor()
    ])

    return transform


def get_dataset(cfg, transform):
    train_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
    train_dataset, valid_dataset = random_split(train_dataset, [int(len(train_dataset) * cfg.data.train_ratio), len(train_dataset)-int(len(train_dataset) * cfg.data.train_ratio)])
    test_dataset = MNIST(root="./data", train=False, transform=transform, download=True)

    return train_dataset, valid_dataset, test_dataset


def get_loaders(cfg, train_dataset, valid_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader



