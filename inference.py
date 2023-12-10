import os

import requests
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import mobilenet_v2

from datasets.dataset import create_dataloader

train_dataset, test_dataset, _, _ = create_dataloader()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resume_model():
    model = mobilenet_v2()

    model.classifier[1] = torch.nn.Linear(1280, len(train_dataset.classes))
    model = model.to(device)

    model.load_state_dict(
        torch.load("weights/model.pt", map_location=torch.device(device))
    )
    model = model.to(device)
    model.eval()
    return model


def transform_data():
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    return transform


@torch.no_grad()
def predict(image):
    model = resume_model()
    transform = transform_data()

    probs = model(transform(image).unsqueeze(0).to(device)).squeeze().softmax(dim=0)

    class_id = probs.argmax().item()
    label = test_dataset.classes[class_id]

    # API for find info about Pokémon
    pokemon = label.lower()
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon}"
    r = requests.get(url)

    print(
        f"class ID: {class_id}, class name: {label}, "
        f"confidence: {100 * probs[class_id].item():.2f}%"
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
