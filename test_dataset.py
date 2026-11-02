from dataset import get_dataloaders
for i in range(6):
    train_loader, test_loader = get_dataloaders(batch_size=32)
    print(f"Iteration {i+1}: Train Loader Length: {len(train_loader)}, Test Loader Length: {len(test_loader)}") 