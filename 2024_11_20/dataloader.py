from torchvision.datasets import MNIST
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch


def get_transform():
    transform = v2.Compose([
        v2.RandomAffine(
            degrees=5,
            translate=(0.1, 0.1)
        ),  # ±5도 기울이기, ±10% 움직이기
        v2.ToImage(),  # PIL -> tv_tensor.Image
        v2.ToDtype(dtype=torch.float32, scale=True)  # dtype을 float로 변환 및 0~1의 사이값으로 scale 해줌
    ])

    return transform


def get_dataset(cfg, transform):
    train_dataset = MNIST(root="./data", train=True, transform=transform, download=True)  # get train dataset
    train_dataset, valid_dataset = random_split(train_dataset, [int(len(train_dataset) * cfg.data.train_ratio), len(train_dataset)-int(len(train_dataset) * cfg.data.train_ratio)])  # split train, valid dataset
    test_dataset = MNIST(root="./data", train=False, transform=transform, download=True)  # get test dataset

    return train_dataset, valid_dataset, test_dataset


def get_loaders(cfg, train_dataset, valid_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True)  # get train dataloader
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=True)  # get valid dataloader
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False)  # get test dataloader

    return train_loader, valid_loader, test_loader