import os

import matplotlib.animation as animation
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from torchvision import transforms

from src.core import hc

#  TODO: (vsedov) (01:28:44 - 24/11/22): Make this into a dataclass - for core refactoring
path = os.path.dirname(os.path.abspath(__file__))
train_path = f"{os.path.dirname(path)}/animal/"
test_path = f"{os.path.dirname(path)}/animal/test1/"
animal_types = f"{path}/animal/types/"
image_size = 64


class Data(data.Dataset):
    def __init__(self, root=train_path):
        """
        Get Image Path.
        """
        self.root = root
        self.folder = os.path.join(root, "train")

    def create_data(self):
        self.dataset = torchvision.datasets.ImageFolder(
            root=animal_types,
            transform=transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

    def get_dl(self, batch_size=32, shuffle=True, drop_last=True):
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    def view_training_data(self):
        """
        View training data
        Function that will show a batch of training images
        Utalises Matplotlib to show the images and its animation features.
        """
        # Plot some training images
        real_batch = next(iter(self.dataloader))

        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(
            np.transpose(
                vutils.make_grid(real_batch[0].to(hc.DEFAULT_DEVICE)[:64], padding=2, normalize=True).cpu(),
                (1, 2, 0),
            )
        )

        fig = plt.figure()
        ims = []
        for i in range(10):
            real_batch = next(iter(self.dataloader))
            ims.append(
                [
                    plt.imshow(
                        np.transpose(
                            vutils.make_grid(real_batch[0].to(hc.DEFAULT_DEVICE)[:64], padding=2, normalize=True).cpu(),
                            (1, 2, 0),
                        ),
                        animated=True,
                    )
                ]
            )
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        ani.save(f"{path}/dynamic_images.mp4")
        plt.show()
