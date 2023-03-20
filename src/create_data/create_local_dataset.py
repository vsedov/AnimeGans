import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.core import hc


class AttrDataset(Dataset):
    def __init__(
        self, csv_file, root_dir, transform=None, hair_cls=12, eye_cls=11
    ):
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
        self.hair_class = hair_cls
        self.eye_class = eye_cls

    def __len__(self):
        return len(self.attr_list)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.root_dir, str(self.attr_list.iloc[idx, 0]) + ".jpg"
        )

        image = Image.open(img_name).convert("RGB")
        image = transforms.Resize((64, 64))(
            image
        )  # <-- Add this line : resize image due1 to memory error

        attrs = self.attr_list.iloc[idx, 1:].astype(float).values

        if self.transform:
            image = self.transform(image)
        return (
            image,
            torch.FloatTensor(attrs[0 : self.hair_class]),  # hair_cls = 8
            torch.FloatTensor(
                attrs[24 - (self.eye_class + 1) :]
            ),  # eye_cls = 8 but because we have 12 total, we do 12 + (12 - 8) -1
        )


def get_dataset(csv_file, root_dir, transform, hair_cls=12, eye_cls=11):
    """
    This function takes in three arguments: csv_file, root_dir, and transform.
    It returns an instance of the AttrDataset class with the given arguments.
    """
    return AttrDataset(csv_file, root_dir, transform, hair_cls, eye_cls)


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


def generate_dataset(hair_classes=12, eye_classes=11):
    path_data = f"{hc.DIR}con/"
    transform_anime = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return get_dataset(
        f"{hc.DIR}create_data/con.csv",
        path_data,
        transform_anime,
        hair_classes,
        eye_classes,
    )


def generate_train_loader(
    # generated_dataset=generate_dataset(),
    hair_classes=12,  # 12
    eye_classes=11,  # 11
    batch_size=64,
    num_workers=16,
    shuffle=True,
    drop_last=True,
):
    generated_dataset = generate_dataset(hair_classes, eye_classes)
    return get_dataloader(
        generated_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
    )


import glob
import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def generate_image_cluster_tags(hair_classes=12, eye_classes=11):
    """
    This function generates a visualization of image clusters based on their tags.
    It uses the t-SNE algorithm to reduce the dimensionality of the tag data and
    the KMeans algorithm to cluster the images based on their tags.

    Args:
        hair_classes (int): The number of hair color classes.
        eye_classes (int): The number of eye color classes.
    """
    # Load the dataset and extract the tag data
    dataset = generate_dataset(hair_classes, eye_classes)
    tag_data = np.array(
        [x[1:] for x in dataset.attr_list.values], dtype=np.float32
    )

    # Perform t-SNE dimensionality reduction on the tag data
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    tag_embedded = tsne.fit_transform(tag_data)

    # Perform KMeans clustering on the tag data
    kmeans = KMeans(n_clusters=3)
    cluster_labels = kmeans.fit_predict(tag_data)

    # Load and standardize the images into a numpy array
    image_folder = "../con/"
    image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
    image_data = []
    for image_file in image_files:
        image = Image.open(image_file)
        image = image.convert("RGB")
        image = np.array(image.resize((64, 64)))
        image = image.astype("float32") / 255.0
        image_data.append(image)
    image_data = np.array(image_data)

    # Visualize the clustered images
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        tag_embedded[:, 0], tag_embedded[:, 1], c=cluster_labels, alpha=0.5
    )
    handles, _ = scatter.legend_elements(num=3)
    legend = ax.legend(
        handles,
        ["Cluster 0", "Cluster 1", "Cluster 2"],
        loc="upper right",
        fontsize="small",
    )
    ax.add_artist(legend)

    # Add image thumbnails to the plot
    image_size = 0.05
    for i, (x, y) in enumerate(tag_embedded):
        thumbnail = plt.axes(
            [x - image_size / 2, y - image_size / 2, image_size, image_size]
        )
        thumbnail.imshow(image_data[i])
        thumbnail.set_xticks([])
        thumbnail.set_yticks([])
        thumbnail.set_frame_on(False)

    plt.savefig("image_clusters.png")


def validate_data_loader(train_loader):

    for step, (real, hair, eye) in enumerate(train_loader):
        print(hair.shape)
        print(eye.shape)


if __name__ == "__main__":
    # validate_data_loader(generate_train_loader())
    generate_image_cluster_tags()
