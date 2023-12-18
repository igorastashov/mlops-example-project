import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from ds.tracking import plot_losses


def training_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    criterion,
    train_loader,
    device: torch.device,
    tqdm_desc,
):
    train_loss, train_accuracy = 0.0, 0.0
    model.train()

    for _, (images, labels) in enumerate(tqdm(train_loader, desc=tqdm_desc), 1):
        images = images.to(device)  # images: batch_size x num_channels x height x width
        labels = labels.to(device)  # labels: batch_size

        optimizer.zero_grad()
        logits = model(images)  # logits: batch_size x num_classes
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.shape[0]
        accuracy = (logits.argmax(dim=1) == labels).sum().item()
        train_accuracy += accuracy

    train_loss /= len(train_loader.dataset)
    train_accuracy /= len(train_loader.dataset)
    return train_loss, train_accuracy


@torch.no_grad()
def validation_epoch(
    model: nn.Module, criterion, test_loader: DataLoader, device: torch.device, tqdm_desc
):
    test_loss, test_accuracy = 0.0, 0.0
    model.eval()
    for images, labels in tqdm(test_loader, desc=tqdm_desc):
        images = images.to(device)  # images: batch_size x num_channels x height x width
        labels = labels.to(device)  # labels: batch_size
        logits = model(images)  # logits: batch_size x num_classes
        loss = criterion(logits, labels)

        test_loss += loss.item() * images.shape[0]
        test_accuracy += (logits.argmax(dim=1) == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy /= len(test_loader.dataset)
    return test_loss, test_accuracy


def train(
    model: object,
    optimizer: object,
    scheduler: object,
    criterion: object,
    train_loader: object,
    test_loader: object,
    num_epochs: object,
    device: object,
    title: object,
) -> object:
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = training_epoch(
            model,
            optimizer,
            criterion,
            train_loader,
            device,
            tqdm_desc=f'Training {epoch}/{num_epochs}',
        )
        test_loss, test_accuracy = validation_epoch(
            model,
            criterion,
            test_loader,
            device,
            tqdm_desc=f'Validating {epoch}/{num_epochs}',
        )

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        train_accuracies += [train_accuracy]
        test_losses += [test_loss]
        test_accuracies += [test_accuracy]

    plot_losses(train_losses, test_losses, train_accuracies, test_accuracies, title)

    return train_losses, test_losses, train_accuracies, test_accuracies
