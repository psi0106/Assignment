import logging
import torch
import torch.nn as nn
import torch.optim as optim
import hydra
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter


def get_transform():
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=False),
    ])

    return transform


def get_dataset(config, transform=None):
    train_dataset = MNIST(root="./data", train=True, transform=transform)
    train_dataset, valid_dataset = random_split(train_dataset, [int(config.data.train_ratio * len(train_dataset)), len(train_dataset) - int(config.data.train_ratio * len(train_dataset))])
    
    test_dataset = MNIST(root="./data", train=False, transform=transform)

    return train_dataset, valid_dataset, test_dataset


def get_loaders(config, train_dataset, valid_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, config.data.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, config.data.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, config.data.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        return x


def train(model, train_loader, valid_loader, criterion, optimizer, epochs, writer):
    best_accuracy = 0.0
    best_model_weights = None
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        writer.add_scalar('Accuracy/validation', accuracy, epoch)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_weights = model.state_dict().copy()

        logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

        model.train()

    if best_model_weights:
        model.load_state_dict(best_model_weights)
        torch.save(best_model_weights, 'best_model.pth')
        logging.info("Best model weights saved with accuracy: {:.2f}%".format(best_accuracy))


def evaluate(model, test_loader, writer):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    logging.info(f'Accuracy: {accuracy:.2f}%')
    writer.add_scalar('Accuracy/test', accuracy)

    writer.close()


@hydra.main(version_base=None, config_path="./config", config_name="train")
def main(cfg):
    OmegaConf.to_yaml(cfg)

    transform = get_transform()

    train_dataset, valid_dataset, test_dataset = get_dataset(cfg, transform)
    logging.info(f"train_dataset: {len(train_dataset)} | valid_dataset: {len(valid_dataset)} | test_dataset: {len(test_dataset)}")

    train_loader, valid_loader, test_loader = get_loaders(cfg, train_dataset, valid_dataset, test_dataset)
    logging.info(f"train_loader: {len(train_loader)} | valid_loader: {len(valid_loader)} | test_loader: {len(test_loader)}")

    model = SimpleModel()
    logging.info(f"success to load model: {model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    writer = SummaryWriter()

    train(model, train_loader, valid_loader, criterion, optimizer, cfg.train.epoch, writer)
    evaluate(model, test_loader, writer)

if __name__ == "__main__":
    main()
