import os
import pathlib

import requests
import torch
from PIL import Image

from ds.dataset import prepare_test_data
from ds.models import ConvNet


# Hyper parameters
EPOCH_COUNT = 10
LR = 1e-2
MOMENTUM = 0.9

# Data configuration
DATA_DIR = pathlib.Path("data/PokemonData")

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Transform data
TRANSFORM = prepare_test_data()


def resume_model():
    model = ConvNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), LR, MOMENTUM)

    model.load_state_dict(
        torch.load("weights/model.pt", map_location=torch.device(device))
    )
    optimizer.load_state_dict(
        torch.load("weights/optimizer.pt", map_location=torch.device(device))
    )

    model.eval()
    return model


def dataset_labels(class_id):
    classes = sorted(os.listdir(DATA_DIR))
    label = classes[class_id]
    return label


@torch.no_grad()
def predict(image):
    model = resume_model()
    probabilities = (
        model(TRANSFORM(image).unsqueeze(0).to(device)).squeeze().softmax(dim=0)
    )
    class_id = probabilities.argmax().item()
    label = dataset_labels(class_id)

    # API for find info about Pokémon
    pokemon = label.lower()
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon}"
    r = requests.get(url)

    print(
        f"class ID: {class_id}, class name: {label}, "
        f"confidence: {100 * probabilities[class_id].item():.2f}%"
    )

    print("Name: ", r.json()["name"])
    print("Base Experience: ", r.json()["base_experience"])
    print("Height: ", r.json()["height"], "m")
    print("Weight: ", r.json()["weight"], "kg")

    return (image.resize([item // 2 for item in image.size])).show()


def main():
    folder_path = "photo"
    file_name = os.listdir(folder_path)[0]
    file_path = os.path.join(folder_path, file_name)

    image = Image.open(file_path).convert("RGB")
    predict(image)


if __name__ == "__main__":
    main()
