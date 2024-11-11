import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np

# 데이터 로드

transform = transforms.ToTensor()
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model_fn = "./model.pth"

# 단순 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = nn.Sequential(    # 입력을 28x28에서 10개의 클래스로 매핑 -> layer를 늘리고 BatchNorm, DropOut 사용, 활성화 함수 LeakyReLU 사용
            nn.Linear(28*28, 500),
            nn.LeakyReLU(),
            nn.BatchNorm1d(500),
            nn.Dropout(0.3),
            nn.Linear(500, 250),
            nn.LeakyReLU(),
            nn.BatchNorm1d(250),
            nn.Dropout(0.3),
            nn.Linear(250, 10),
            nn.LogSoftmax(dim=1)    #LogSoftmax 사용
            )

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten
        y = self.layer(x)
        return y

# 모델 초기화
model = SimpleModel()

# 손실 함수 및 옵티마이저 정의
criterion = nn.NLLLoss()  #LogSoftmax에 맞는 NLLLoss 사용
optimizer = optim.Adam(model.parameters(), lr=0.01) #lr가 너무 낮게 설정되어있음 0.00001 -> 0.01, SGD는 학습이 많이 필요함 -> Adam 옵티마이저로 바꿔 더욱 효율적으로 학습 그리고 lr가 점점 낮아지기 때문에 조금 크게 잡음

# 훈련
def train(model, train_loader, criterion, optimizer, epochs=1): #epochs를 16
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 평가
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')

# 실행
train(model, train_loader, criterion, optimizer, epochs=16)
evaluate(model, test_loader)
