import torch.nn as nn


class SimpelModel(nn.Module):
    def __init__(self):
        super(SimpelModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)  # input_size = 28 * 28, output_size = 10

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)  # feedforward
        return x


