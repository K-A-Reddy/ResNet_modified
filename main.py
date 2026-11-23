import argparse
import torch
import torch.nn as nn
from models.resnet18 import (
    ImportanceGuidedResNet18,
    LearnedWeightedResNet18,
    ResNet18,
)
from dataset import get_dataloaders
from train import train_one_epoch
from evaluate import evaluate
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("classical", "importance", "learned"),
        default="classical",
        help="Choose classical max-pool, importance-guided max, or learned weighted-mean pooling",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, test_loader = get_dataloaders(batch_size=128)
    model_classes = {
        "classical": ResNet18,
        "importance": ImportanceGuidedResNet18,
        "learned": LearnedWeightedResNet18,
    }
    model = model_classes[args.model](num_classes=10).to(device)
    print(f"Model variant: {args.model}")
    criterion = nn.CrossEntropyLoss()
    learning_rate=0.1
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=0.0001)
    epochs = 164
    counter=0
    for epoch in range(epochs):
        counter+=1
        if counter==91:
            counter=0
            learning_rate=learning_rate*0.1
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
