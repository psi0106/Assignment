import hydra
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from dataloader import get_transform, get_dataset, get_loaders
from model import SimpelModel
import logging
import torch.nn as nn
from trainer import train, evaluate


@hydra.main(version_base=None, config_path="./config", config_name='train')
def main(cfg):
    OmegaConf.to_yaml(cfg)

    transform = get_transform()

    train_dataset, valid_dataset, test_dataset = get_dataset(cfg, transform)
    logging.info(f"train_dataset: {len(train_dataset)}, valid_dataset: {len(valid_dataset)}, test_dataset: {len(test_dataset)}")

    train_loader, valid_loader, test_loader = get_loaders(cfg, train_dataset, valid_dataset, test_dataset)
    logging.info(f"train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}, test_loader: {len(test_loader)}")

    model = SimpelModel()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.train.lr)
    writer = SummaryWriter()

    train(model, train_loader, valid_loader, criterion, optimizer, cfg.train.epochs, writer)
    evaluate(model, test_loader, writer)


if __name__ == '__main__':
    main()
