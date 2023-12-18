import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from config import ConvNetConfig
from ds.dataset import create_dataloader, remove_bed_images
from ds.models import ConvNet
from ds.runner import train


# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


cs = ConfigStore.instance()
cs.store(name="ConfNet_config", node=ConvNetConfig)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: ConvNetConfig):
    print(OmegaConf.to_yaml(cfg))

    # Model and Optimizer
    model_name = "ConvNet"
    model = ConvNet().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.params.lr, momentum=cfg.params.momentum
    )
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=cfg.params.epoch_count
    )

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
