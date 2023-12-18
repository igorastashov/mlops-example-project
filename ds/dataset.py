import glob
import os
from pathlib import Path
from typing import Any

import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def remove_bed_images(data_dir: Path):
    bad_images = glob.glob(f"{data_dir}/*/*.svg")
    for bad_image in bad_images:
        os.remove(bad_image)


TRAIN_NORMALIZED_MEAN = [0.485, 0.456, 0.406]
TRAIN_NORMALIZED_STDEV = [0.229, 0.224, 0.225]
NORMALIZE = T.Normalize(mean=TRAIN_NORMALIZED_MEAN, std=TRAIN_NORMALIZED_STDEV)


def prepare_train_data():
    """Add Augmentations"""
    train_transform = T.Compose(
        [
            T.RandomResizedCrop(size=224, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            NORMALIZE,
        ]
    )
    return train_transform


def prepare_test_data():
    test_transform = T.Compose(
        [
            T.Resize(size=256),
            T.CenterCrop(size=224),
            T.ToTensor(),
            NORMALIZE,
        ]
    )
    return test_transform


class PokemonDataset(Dataset[Any]):
    SPLIT_RANDOM_SEED = 42
    TEST_SIZE = 0.25

    def __init__(self, root, train=True, load_to_ram=False, transform=None):
        super().__init__()
        self.root = root
        self.train = train
        self.load_to_ram = load_to_ram
        self.transform = transform
        self.to_tensor = T.ToTensor()
        self.all_files = []
        self.all_labels = []
        self.images = []

        self.classes = sorted(os.listdir(self.root))
        for i, class_name in tqdm(enumerate(self.classes), total=len(self.classes)):
            files = sorted(os.listdir(os.path.join(self.root, class_name)))
            train_files, test_files = train_test_split(
                files, random_state=self.SPLIT_RANDOM_SEED + i, test_size=self.TEST_SIZE
            )
            if self.train:
                self.all_files += train_files
                self.all_labels += [i] * len(train_files)
                if self.load_to_ram:
                    self.images += self._load_images(train_files, i)

            else:
                self.all_files += test_files
                self.all_labels += [i] * len(test_files)
                if self.load_to_ram:
                    self.images += self._load_images(test_files, i)

    def _load_images(self, image_files, label):
        images = []
        for filename in image_files:
            image = Image.open(
                os.path.join(self.root, self.classes[label], filename)
            ).convert("RGB")
            images += [image]

        return images

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, item):
        label = self.all_labels[item]
        if self.load_to_ram:
            image = self.images[item]
        else:
            filename = self.all_files[item]
            image = Image.open(
                os.path.join(self.root, self.classes[label], filename)
            ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def create_dataloader(
    root: Path,
    batch_size: int,
    load_to_ram: bool = False,
    pin_memory: bool = True,
    num_workers: int = 2,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_dataset = PokemonDataset(
        root=root,
        train=True,
        load_to_ram=load_to_ram,
        transform=prepare_train_data(),
    )

    test_dataset = PokemonDataset(
        root=root,
        train=False,
        load_to_ram=load_to_ram,
        transform=prepare_test_data(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    return train_loader, test_loader
