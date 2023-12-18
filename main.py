import pathlib

import torch

from ds.dataset import create_dataloader, remove_bed_images
from ds.models import ConvNet
from ds.runner import train


# Hyper parameters
EPOCH_COUNT = 10
LR = 1e-2
MOMENTUM = 0.9
BATCH_SIZE = 32

# Data configuration
DATA_DIR = pathlib.Path("data/PokemonData")

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    # Model and Optimizer
    model_name = "ConvNet"
    model = ConvNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), LR, MOMENTUM)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCH_COUNT)

    # Remove bad images
    remove_bed_images(DATA_DIR)

    # Create the data loaders
    train_loader, test_loader = create_dataloader(
        root=DATA_DIR,
        batch_size=BATCH_SIZE,
        load_to_ram=False,
        pin_memory=True,
        num_workers=2,
    )

    # Run epochs
    train(
        model,
        optimizer,
        scheduler,
        criterion,
        train_loader,
        test_loader,
        num_epochs=EPOCH_COUNT,
        device=device,
        title=model_name,
    )

    # Save model and optimizer
    torch.save(model.state_dict(), "weights/model.pt")
    torch.save(optimizer.state_dict(), "weights/optimizer.pt")


if __name__ == "__main__":
    main()
