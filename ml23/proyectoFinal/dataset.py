# dataset.py

import pathlib
from typing import Any, Callable, Optional
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
import numpy as np
import json

# Dictionary of animals 90
ANIMALS_MAP = {
    0: "antelope", 1: "badger", 2: "bat", 3: "bear", 4: "bee", 5: "beetle", 6: "bison", 7: "boar", 8: "butterfly",
    9: "cat", 10: "caterpillar", 11: "chimpanzee", 12: "cockroach", 13: "cow", 14: "coyote", 15: "crab", 16: "crow",
    17: "deer", 18: "dog", 19: "dolphin", 20: "donkey", 21: "dragonfly", 22: "duck", 23: "eagle", 24: "elephant",
    25: "flamingo", 26: "fly", 27: "fox", 28: "goat", 29: "goldfish", 30: "goose", 31: "gorilla", 32: "grasshopper",
    33: "hamster", 34: "hare", 35: "hedgehog", 36: "hippopotamus", 37: "hornbill", 38: "horse", 39: "hummingbird",
    40: "hyena", 41: "jellyfish", 42: "kangaroo", 43: "koala", 44: "ladybugs", 45: "leopard", 46: "lion", 47: "lizard",
    48: "lobster", 49: "mosquito", 50: "moth", 51: "mouse", 52: "octopus", 53: "okapi", 54: "orangutan", 55: "otter",
    56: "owl", 57: "ox", 58: "oyster", 59: "panda", 60: "parrot", 61: "pelecaniformes", 62: "penguin", 63: "pig",
    64: "pigeon", 65: "porcupine", 66: "possum", 67: "raccoon", 68: "rat", 69: "reindeer", 70: "rhinoceros",
    71: "sandpiper", 72: "seahorse", 73: "seal", 74: "shark", 75: "sheep", 76: "snake", 77: "sparrow", 78: "squid",
    79: "squirrel", 80: "starfish", 81: "swan", 82: "tiger", 83: "turkey", 84: "turtle", 85: "whale", 86: "wolf",
    87: "wombat", 88: "woodpecker", 89: "zebra"
}

file_path = pathlib.Path(__file__).parent.absolute()

# Transformaciones
# DATA TRANSFORMATION FOR TRAINING AND VAL.
# NO DATA AUGMENTATION YET
def get_transforms(split, img_size):
    common = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((img_size, img_size))
    ]

    mean, std = 0.5, 0.5
    if split == "train":
        transforms = torchvision.transforms.Compose([
            *common,
            torchvision.transforms.Normalize((mean,), (std,))
        ])
    else:
        transforms = torchvision.transforms.Compose([
            *common,
            torchvision.transforms.Normalize((mean,), (std,))
        ])

    # For visualization
    de_normalize = UnNormalize(mean=[mean], std=[std])
    return transforms, de_normalize

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

# DataLoader para el conjunto de datos
def get_loader(split, batch_size, shuffle=True, num_workers=0):
    dataset = AnimalDataset(root=file_path, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataset, dataloader

# Clase del conjunto de datos
class AnimalDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.img_size = 64  # Updated image size
        self.target_transform = target_transform
        self.split = split
        self.root = root
        self.un_normalize = None
        self.transform, self.un_normalize = get_transforms(
            split=self.split,
            img_size=self.img_size
        )

        df = self._read_data()
        _str_to_array = [np.fromstring(val, dtype=int, sep=' ') for val in df['pixels'].values]

        self._samples = np.array(_str_to_array)
        if split == "test":
            self._labels = np.empty(shape=len(self._samples))
        else:
            self._labels = df['emotion'].values

    def _read_data(self):
        base_folder = pathlib.Path(self.root) / "data"
        _split = "train" if self.split == "train" or "val" else "test"
        file_name = f"{_split}.csv"
        data_file = base_folder / file_name

        if not os.path.isfile(data_file.as_posix()):
            raise RuntimeError(
                f"{file_name} not found in {base_folder} or corrupted. "
                f"You can download it from the dataset source."
            )

        df = pd.read_csv(data_file)
        if self.split != "test":
            train_val_split = json.load(open(base_folder / "split.json", "r"))
            split_samples = train_val_split[self.split]
            df = df.iloc[split_samples]
        return df

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        _vector_img = self._samples[idx]

        sample_image = _vector_img.reshape(self.img_size, self.img_size).astype('uint8')
        if self.transform is not None:
            image = self.transform(sample_image)
        else:
            image = torch.from_numpy(sample_image)

        target = self._labels[idx]
        animal = ANIMALS_MAP[target]
        if self.target_transform is not None:
            target = self.target_transform(target
