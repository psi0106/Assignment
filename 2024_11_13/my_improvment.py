import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import hydra
from torchvision.transforms import v2
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split

# 어떤 데이터 증강을 할 것인지 정의하는 함수
def get_transform():
    transform = v2.Compose([
        v2.RandomAffine(
            degrees=0,  # 회전 X
            translate=(3 / 28, 3 / 28),  # shift 연산을 3 픽셀
            scale=None,  # 스케일링 X
            shear=None  # 기울이기 X
        ),
        v2.ToTensor()  # np.array -> torch.tensor
    ])

    return transform


# 단순 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 10)  # 입력을 28x28에서 10개의 클래스로 매핑 레이어 정의

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten
        x = self.fc1(x)  # forward
        return x


# 데이터 다운
def get_dataset(config, transform=None):
    train_data = MNIST(root='./data', train=True, download=True, transform=transform)  # train_dataset을 transform을 적용하여 다운
    train_data, valid_data = random_split(train_data, [int(config.data.train_ratio*len(train_data)), len(train_data) - int(config.data.train_ratio*len(train_data))] )  # train과 valid 나누기
    test_data = MNIST(root='./data', train=False, download=True, transform=transform)  # test_dataset을 transform을 적용하여 다운

    return train_data, valid_data, test_data


# 데이터 로드
def get_loader(config, train_dataset, valid_dataset, test_dataset):

    # batch_size로 dataset을 나누고 매 epoch마다 shuffle
    train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset,batch_size=config.data.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.data.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


# 훈련
def train(model, train_loader, valid_loader, criterion, optimizer, writer, epochs=1):
    model.train()  # batch_norm과 drop_out을 활성화
    for epoch in range(epochs):
        train_iter = 0  # 학습 iter 초기화
        valid_iter = 0  # 검증 iter 초기화

        # 학습
        for images, labels in train_loader:
            outputs = model(images)  # batch_size로 받은 데이터들을 불러온 model에 feedforward
            loss = criterion(outputs, labels)  # # 불러온 loss_function으로 loss 계산
            train_iter += 1  # iter 매 학습마다 1 더하기
            logging.info(f'Iter [{train_iter}/], Loss: {loss.item():.4f}')  # 매 학습마다 loss 출력
            writer.add_scalar("Loss/train", loss, train_iter)  # 매 학습마다 loss 저장
            optimizer.zero_grad()  # 기울기 초기화
            loss.backward()  # 백프로파게이션으로 gradient 계산
            optimizer.step()  # learning_late 만큼 이동

        # 검증
        model.eval()  # batch_norm과 drop_out을 비활성화
        with torch.no_grad():  # 체인룰을 위한 계산을 없애기 위한 코드
            for images, labels in valid_loader:  # test_loader에서 images와 labels를 batch_size만큼 보내준다.
                outputs = model(images)  # batch_size로 받은 데이터들을 불러온 model에 feedforward
                loss = criterion(outputs, labels)  # 불러온 loss_function으로 loss 계산
                valid_iter += 1
                logging.info(f'Iter [{valid_iter}/], Loss: {loss.item():.4f}')  # valid_iter에 따른 loss 출력
                writer.add_scalar("Loss/valid", loss, valid_iter)  # writer에 valid_iter에 따른 loss 저장

        model.train()  # 다시 학습을 위해 학습 모드로 변환


# 평가
def evaluate(model, test_loader, writer):
    model.eval()  # batch_norm과 drop_out을 비활성화
    correct, total = 0, 0  # 맞은 개수와 전체 데이터 개수 초기화
    with torch.no_grad():  # 체인룰을 위한 계산을 없애기 위한 코드
        for images, labels in test_loader:  # test_loader에서 images와 labels를 batch_size만큼 보내준다.
            outputs = model(images)  # batch_size로 받은 데이터들을 불러온 model에 feedforward
            _, predicted = torch.max(outputs, 1)  # torch.max는 가장 큰 value와 index를 반환하기 때문에 value는 필요 없어서 _로 받아주고 우리가 필요한 index를 predicted 변수로 받기
            total += labels.size(0)  # 전체 데이터 개수를 저장해 놓은 변수에 batch_size만큼 더하기
            correct += (predicted == labels).sum().item()  # 예측된 인덱스가 라벨과 같으지 확인 후 맞은 개수에 더하기
    logging.info(f'Accuracy: {100 * correct / total:.2f}%')  # 정확도 출력
    accuracy = 100 * correct / total  # 맞은 데이터 / 전체 데이터 해서 정확도 측정
    writer.add_scalar("Loss/test", accuracy)  # test accuracy 저장


# main 코드
@hydra.main(version_base=None, config_path='./config', config_name='my_train')  # hydra 버전은 None으로 할당, config 파일 위치와 이름 할당
def main(cfg):
    OmegaConf.to_yaml(cfg)  # config 불러 오기

    transform = get_transform()  # transform 불러 오기

    train_dataset, valid_dataset, test_dataset = get_dataset(cfg, transform)  # dataset을 trasform을 적용시켜 다운 및 config 전달하여 train_ratio 전달
    logging.info(f"Train_dataset:{len(train_dataset)}, Valid_dataset:{len(valid_dataset)}, Test_dataset:{len(test_dataset)}")  # dataset 확인

    train_loader, valid_loader, test_loader = get_loader(cfg, train_dataset, valid_dataset, test_dataset)  # 다운 받은 dataset을 get_loader로 보내 학습에 적합한 dataloader 형태로 변환
    logging.info(f"Train_loader:{len(test_loader)}, Valid_loader:{len(valid_loader)}, Test_loader:{len(test_dataset)}")  # dataloader 확인

    model = SimpleModel()  # 모델 불러 오기
    logging.info(f"success to load model: {model}")  # 모델 확인

    criterion = nn.CrossEntropyLoss()  # loss_function 정의
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)  # optimizer 정의 및 하이퍼 파라미터인 learning_late config에서 불러오기
    writer = SummaryWriter()  # Tensorboard로 저장하기 위해 SummaryWriter 불러 오기

    train(model, train_loader, valid_loader, criterion, optimizer, writer, epochs=cfg.train.epoch)  # 학습 및 검증
    evaluate(model, test_loader, writer)  # 테스트


if __name__ == "__main__":
    main()
