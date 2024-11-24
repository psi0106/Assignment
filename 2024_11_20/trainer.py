import torch
import logging


def train(model, train_loader, valid_loader, criterion, optimizer, epochs, writer):
    best_accuracy = 0.0  # best accuracy 초기화
    best_model_weights = None  # best model weights 초기화
    model.train()  # batch_norm, drop_out 활성화

    for epoch in range(epochs):
        running_loss = 0.0  # loss 초기화

        for images, labels in train_loader:
            outputs = model(images)  # images feedforward
            loss = criterion(outputs, labels)  # loss 계산

            optimizer.zero_grad()  # gradient 초기화
            loss.backward()  # backpropagation
            optimizer.step()  # step

            running_loss += loss.item()  # sum of loss

        avg_train_loss = running_loss / len(train_loader)  # loss avg
        writer.add_scalar('Loss/train', avg_train_loss, epoch)  # tensorboard에 저장

        model.eval()  # batch_norm, drop_out 비활성화
        correct, total = 0, 0  # correct, total 초기화
        with torch.no_grad():  # 역전파와 gradient 계산 비활성화
            for images, labels in valid_loader:
                outputs = model(images)  # images feedforward
                _, predicted = torch.max(outputs, 1)  # 확률 값이 가장 큰 index 값 저장
                total += labels.size(0)  # 데이터의 개수 저장
                correct += (predicted == labels).sum().item()  # 맞은 데이터 개수 저장

        accuracy = 100 * correct / total  # 정확도 계산
        writer.add_scalar('Accuracy/validation', accuracy, epoch)  # epoch에 따른 accracy 저장

        if accuracy > best_accuracy:  # best accuracy 보다 현 epoch의 accuracy가 높다면
            best_accuracy = accuracy  # best accuracy로 저장
            best_model_weights = model.state_dict().copy()  # best accuracy의 model weights 저장

        logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

        model.train()

    if best_model_weights:
        model.load_state_dict(best_model_weights)  # model에 best accuracy의 weight를 가져옴
        torch.save(best_model_weights, 'best_model.pth')  # model 저장
        logging.info(f"Best model weights saved with accuracy: {best_accuracy:.2f}")


def evaluate(model, test_loader, writer):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)  # images feedforward
            _, predicted = torch.max(outputs, 1)  # 확률값이 가장큰 index 저장
            total += labels.size(0)  # 데이터의 개수 저장
            correct += (predicted == labels).sum().item()  # 맞은 데이터 개수 저장
    accuracy = 100 * correct / total  # 정확도 계산
    logging.info(f"Accuracy: {accuracy:.2f}%")
    writer.add_scalar('Accuracy/test', accuracy)  # epoch에 따른 accracy 저장

    writer.close()  # tensorboard writer close
