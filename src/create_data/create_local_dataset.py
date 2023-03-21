import glob
import os

import numpy as np
import pandas as pd
import torch
import umap
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE
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


def generate_image_cluster_tags(
    hair_classes=12, eye_classes=11, max_images=50, mini_batch=False
):
    # Load the dataset and extract the tag data
    dataset = generate_dataset(hair_classes, eye_classes)
    tag_data = np.array(
        [x[1:] for x in dataset.attr_list.values], dtype=np.float32
    )

    # Perform t-SNE dimensionality reduction on the tag data
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    tag_embedded = tsne.fit_transform(tag_data)

    # Cluster the tag data using KMeans or MiniBatchKMeans
    if mini_batch:
        kmeans = MiniBatchKMeans(n_clusters=3, batch_size=1000)
    else:
        kmeans = KMeans(n_clusters=3)
    cluster_labels = kmeans.fit_predict(tag_data)

    # Load and standardize the images into a numpy array
    image_folder = "../con"
    image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
    if max_images is not None:
        if max_images > len(image_files):
            max_images = len(image_files)
        image_files = image_files[:max_images]

    # Create a figure to hold the plot
    fig = plt.figure(figsize=(16, 10))

    # Create a grid to hold the images and tags
    grid = ImageGrid(fig, 111, nrows_ncols=(10, 20), axes_pad=0.1)

    # Add the images to the plot
    image_data = []
    for i, image_file in enumerate(image_files):
        if i >= max_images:
            break

        image = Image.open(image_file)
        image = image.convert("RGB")
        image = np.array(image.resize((64, 64)))
        image = image.astype("float32") / 255.0
        image_data.append(image)

        # Add the image thumbnail to the grid
        ax = grid[i]
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])

    # Add the cluster labels to the plot
    cluster_colors = ["r", "g", "b"]
    cluster_labels_str = ["Cluster 0", "Cluster 1", "Cluster 2"]
    for i in range(len(cluster_labels_str)):
        ax = grid[max_images + i]
        ax.axis("off")
        ax.add_artist(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=cluster_labels_str[i],
                markerfacecolor=cluster_colors[i],
                markersize=10,
            )
        )
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3)

    # Show the plot
    plt.show()


def validate_data_loader(train_loader):

    for step, (real, hair, eye) in enumerate(train_loader):
        print(hair.shape)
        print(eye.shape)


# Function to load images from the given directory
def load_images(directory, resize_shape):
    image_list = []
    for filepath in glob.glob(os.path.join(directory, "*.jpg")):
        img = Image.open(filepath)
        img_resized = img.resize(resize_shape)
        img_array = np.array(img_resized).flatten()
        image_list.append(img_array)
    return np.array(image_list)


def visualize_umap(image_directory, resize_shape):
    images = load_images(image_directory, resize_shape)

    print("Fitting UMAP Model...")
    reducer = umap.UMAP(n_components=2, random_state=42, min_dist=1.0)
    embedded_images = reducer.fit_transform(images)

    fig, ax = plt.subplots(figsize=(64, 64))
    for i, (x, y) in enumerate(embedded_images):
        image = images[i].reshape(resize_shape[0], resize_shape[1], -1)
        im = OffsetImage(image, zoom=1, cmap="gray")
        ab = AnnotationBbox(im, (x, y), xycoords="data", frameon=False)
        ax.add_artist(ab)

    ax.set_xlim(
        embedded_images[:, 0].min() - 10, embedded_images[:, 0].max() + 10
    )
    ax.set_ylim(
        embedded_images[:, 1].min() - 10, embedded_images[:, 1].max() + 10
    )
    plt.title("UMAP Visualization of Image Dataset")
    plt.axis("off")
    plt.savefig("umap.png")


# Function to create UMAP visualization
if __name__ == "__main__":
    # generate_image_cluster_tags()
    # Example usage
    # image_directory = "../con/"
    # resize_shape = (32, 32)
    # visualize_umap(image_directory, resize_shape)
    pass
