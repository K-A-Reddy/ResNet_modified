import torch


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images,labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        predicted=torch.argmax(outputs,dim=1)
        correct+=(predicted==labels).sum().item()
        total+=labels.size(0)
    if total == 0:
        return 0.0, 0.0
    average_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return average_loss, accuracy
