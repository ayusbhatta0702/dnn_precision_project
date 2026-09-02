"""
Dataset loaders for CIFAR-10, CIFAR-100 and MNIST via torchvision.
Downloads to `root` automatically on first use (needs internet once).
"""
import torch
from torchvision import datasets, transforms


CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)


def get_cifar10_loaders(root, batch_size=128, num_workers=2):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                num_workers=num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers)
    return train_loader, test_loader


def get_cifar100_loaders(root, batch_size=128, num_workers=2):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train_set = datasets.CIFAR100(root=root, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR100(root=root, train=False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                num_workers=num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers)
    return train_loader, test_loader


def get_mnist_loaders(root, batch_size=128, num_workers=2):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    train_set = datasets.MNIST(root=root, train=True, download=True, transform=tf)
    test_set = datasets.MNIST(root=root, train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                num_workers=num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers)
    return train_loader, test_loader


LOADER_BUILDERS = {
    "cifar10": get_cifar10_loaders,
    "cifar100": get_cifar100_loaders,
    "mnist": get_mnist_loaders,
}
