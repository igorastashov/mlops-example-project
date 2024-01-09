import hydra
import torch
from omegaconf import DictConfig

from ds.dataset import create_dataloader, remove_bed_images
from ds.models import ConvNet
from ds.runner import train


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Model and Optimizer
    model_name = cfg.model.name
    model = ConvNet().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.params.lr, momentum=cfg.params.momentum
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=cfg.params.epoch_count
    )

    criterion = torch.nn.CrossEntropyLoss()

    # Remove bad images
    remove_bed_images(cfg.paths.data)

    # Create the data loaders
    train_loader, test_loader = create_dataloader(
        root_path=cfg.paths.data,
        batch_size=cfg.params.batch_size,
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
        num_epochs=cfg.params.epoch_count,
        device=device,
        title=model_name,
    )

    # Save model and optimizer
    torch.save(model.state_dict(), "weights/model.pt")
    torch.save(optimizer.state_dict(), "weights/optimizer.pt")


if __name__ == "__main__":
    main()
