import torch
import torch.nn as nn
from models.resnet18 import ResNCIF18
from dataset import get_dataloaders
from train import train_one_epoch
from evaluate import evaluate
def main():
    device = torch.device("cuda" if torch.backends.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, test_loader = get_dataloaders(batch_size=128)
    model = ResNCIF18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    learning_rate=0.1
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=0.0001)
    epochs = 164
    counter=0
    for epoch in range(epochs):
        counter+=1
        if counter==91:
            counter=0
            learing_rate=learning_rate*0.1
            optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=0.0001)
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Accuracy: {test_accuracy:.2f}%"
        )
if __name__ == "__main__":
    main()