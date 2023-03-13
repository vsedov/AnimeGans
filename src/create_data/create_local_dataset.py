import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.core import hc


class AttrDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        This class is a custom PyTorch dataset that reads and processes the data.
        It takes in two required arguments: csv_file and root_dir. The csv_file argument
        is a string that specifies the path to the csv file that contains the features
        for the images. The root_dir argument is a string that specifies the directory
        where all the images are stored. An optional argument transform can be passed
        to this class, which is a callable function that can be used to apply any
        transformations to the images.

        Args:
            csv_file (string): Path to the csv file with featrues.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.attr_list = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.attr_list)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.root_dir, str(self.attr_list.iloc[idx, 0]) + ".png"
        )
        image = Image.open(img_name).convert("RGB")
        attrs = self.attr_list.iloc[idx, 1:].astype(float).values

        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.FloatTensor(attrs[0:12]),
            torch.FloatTensor(attrs[12:]),
        )


def get_dataset(csv_file, root_dir, transform):
    """
    This function takes in three arguments: csv_file, root_dir, and transform.
    It returns an instance of the AttrDataset class with the given arguments.
    """
    return AttrDataset(csv_file, root_dir, transform)


def get_dataloader(dataset, batch_size, num_workers, shuffle, drop_last):
    """
    Parameters:
        - dataset (AttrDataset): An instance of the AttrDataset class.
        - batch_size (int): The number of samples to be loaded in each batch.
        - num_workers (int): The number of workers to use for loading the data in parallel.
        - shuffle (bool): A boolean that specifies whether the data should be shuffled before loading.
        - drop_last (bool): A boolean that specifies whether the last batch should be dropped if its size is smaller than batch_size.

    Returns:
        - PyTorch DataLoader object: Can be used to load the data in batches.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def validate_data_loader(train_loader):

    for step, (real, ahir, eye) in enumerate(train_loader):
        # Get the hair color and eye color for each image
        hair_color = np.argmax(ahir.numpy(), axis=1)
        eye_color = np.argmax(eye.numpy(), axis=1)
        hair_colors = [
            "orange",
            "white",
            "aqua",
            "gray",
            "green",
            "red",
            "purple",
            "pink",
            "blue",
            "black",
            "brown",
            "blonde",
        ]
        eye_colors = [
            "gray",
            "black",
            "orange",
            "pink",
            "yellow",
            "aqua",
            "purple",
            "green",
            "brown",
            "red",
            "blue",
        ]
        # Plot the first 10 images
        fig = plt.figure(figsize=(5, 5))
        for i in range(10):
            ax = fig.add_subplot(5, 5, i + 1)
            ax.imshow(np.array(real[i].permute(1, 2, 0)))
            ax.axis("off")
            ax.set_title(
                f"Hair: {hair_colors[hair_color[i]]}, Eye: {eye_colors[eye_color[i]]}"
            )
        plt.tight_layout()
        plt.show()
        break


path_data = f"{hc.DIR}/data/"

transform_anime = transforms.Compose(
    [
        transforms.ToTensor(),
        # better results if you had normalized this data instead?
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


generated_dataset = get_dataset(
    f"{hc.DIR}/create_data/features.csv", path_data, transform_anime
)

train_loader = get_dataloader(
    generated_dataset,
    batch_size=64,
    num_workers=16,
    shuffle=True,
    drop_last=True,
)

if __name__ == "__main__":
    validate_data_loader(train_loader)
