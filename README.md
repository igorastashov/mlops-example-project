## MLops project

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Build Status](https://github.com/igorastashov/MLops-project/actions/workflows/checks.yml/badge.svg)](https://github.com/igorastashov/MLops-project/actions/workflows/checks.yml)

**Pokemon image classification using Mobilenet_v2 model**

Astashov I.V., 2023.

This repository contains model, evaluation code and training code on dataset
from [kaggle](https://www.kaggle.com/datasets/lantian773030/pokemonclassification).
**If you would like to run pretrained model on your image see [(2) Quick start](https://github.com/igorastashov/MLops-project#2-quick-start)**.

## (1) Setup

### Install packages

- This template use poetry to manage dependencies of your project.
  First you need to [install poetry](https://python-poetry.org/docs/#installing-with-pipx);
- Next run `poetry install` and `poetry shell`.

## (2) Quick start

### Download model weights

```
# Download model weights
cd weights
bash download_weights.sh
cd ../..
```

### Run on a single image

This command runs the model on a single image, and outputs the prediction.
Put your Pokémon image into the appropriate folder `photo` to test.

```
# Model weights need to be downloaded
python inference.py
```

## (3) Dataset

### Download the dataset

A script for downloading the dataset is as follows:

```
# Download the dataset
cd data
bash download_data.sh
cd ../..
```

All the 150 Pokémon included in this dataset are from generation one.
There are around 25 - 50 images for each Pokémon.
All of them with the Pokémon in the center.
Most (if not all) of the images have relatively high quality (correct labels, centered).
The images don't have extremely high resolutions so it's perfect for some light deep learning.

If the script doesn't work, an alternative will be to download the zip files manually
from the [link](https://www.kaggle.com/datasets/lantian773030/pokemonclassification/download?datasetVersionNumber=1).
One can place the dataset zip files in `data`, respectively, and then unzip the zip file to set everything up.

**PAY ATTENTION**

**This repository runs Data Version Control (DVC) for training and validation data.
Pre-configured Google Drive remote storage stores raw input data.**

```console
$ dvc remote list
my_remote gdrive://1RXz3Mv7OxVveHtQ7c1ZtGgazDh6bPFJz
```

You can run `dvc pull` to download the data:

```console
$ dvc pull
```

## (4) Train and Evaluation model

Example script to train and evaluate model.

```
# Train MobileNet_V2
python train.py
```

## (A) Acknowledgments

This repository borrows partially from [Isadrtdinov](https://github.com/isadrtdinov/intro-to-dl-hse/blob/2022-2023/seminars/201/seminar_04.ipynb), and [FUlyankin](https://github.com/FUlyankin/deep_learning_pytorch/tree/main/week08_fine_tuning) repositories.
Repository design taken from [v-goncharenko](https://github.com/v-goncharenko/data-science-template) and [PeterWang512](https://github.com/PeterWang512/CNNDetection).
