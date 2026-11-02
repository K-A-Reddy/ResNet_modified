import torch
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    # Disable gradient calculation
    with torch.no_grad():
        for images, labels in data_loader:
            # Move data to the selected device
            images = images.to(device)
            labels = labels.to(device)
            # Forward propagation
            outputs = model(images)
            # Calculate loss
            loss = criterion(outputs, labels)
            # Track loss
            total_loss += loss.item() * images.size(0)
            # Track accuracy
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    average_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return average_loss, accuracy