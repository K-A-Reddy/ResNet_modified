import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(batch_size=128):

    raw_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True
    )

    # Mean image: (3, 32, 32)
    mean_image = torch.tensor(
        raw_dataset.data.astype("float32") / 255.0
    ).mean(dim=0).permute(2, 0, 1)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_image,
            std=torch.ones_like(mean_image)
        ),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_image,
            std=torch.ones_like(mean_image)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader