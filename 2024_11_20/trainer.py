import torch
import logging


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

        logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

        model.train()

    if best_model_weights:
        model.load_state_dict(best_model_weights)
        torch.save(best_model_weights, 'best_model.pth')
        logging.info(f"Best model weights saved with accuracy: {best_accuracy:2f}")


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
    logging.info(f"Accuracy: {accuracy:.2f}%")
    writer.add_scalar('Accuracy/test', accuracy)

    writer.close()