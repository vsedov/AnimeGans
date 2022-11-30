import os
import shutil

import matplotlib.animation as animation
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from torchvision import transforms

from src.core import hc


class Data(data.Dataset):
    def __init__(self):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.dirname = os.path.dirname(self.path)
        self.ds_path = None
        self.animal = False

        self.image_size = hc.initial["params"]["image_size"]

    def set_animal_path(self):
        self.ds_path = f"{self.path}/animal/types/"
        self.animal = True
        print(self.ds_path)

    def set_human_path(self):
        self.ds_path = f"{self.path}/human_faces/img_align_celeba/"
        print(self.ds_path)

    def move_data_to_types(self):
        # !mv train/dog.*.jpg types/dog/
        # !mv train/cat.*.jpg types/cat/
        # if dog or cat does not exist, create it
        if not os.path.exists(f"{self.ds_path}/dog"):
            os.makedirs(f"{self.ds_path}/dog")
        if not os.path.exists(f"{self.ds_path}/cat"):
            os.makedirs(f"{self.ds_path}/cat")

        with os.scandir(self.folder) as entries:
            for entry in entries:
                if entry.name.startswith("dog"):
                    shutil.move(entry.path, f"{self.ds_path}/dog/")
                elif entry.name.startswith("cat"):
                    shutil.move(entry.path, f"{self.ds_path}/cat/")
                else:
                    print("No match")

    def create_data(self):
        self.dataset = torchvision.datasets.ImageFolder(
            root=self.ds_path,
            transform=transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

        if self.animal:
            dog_index = self.dataset.class_to_idx["dog"]
            cat_index = self.dataset.class_to_idx["cat"]
            self.dog_dataset = torch.utils.data.Subset(
                self.dataset, [i for i, t in enumerate(self.dataset.targets) if t == dog_index]
            )
            self.cat_dataset = torch.utils.data.Subset(
                self.dataset, [i for i, t in enumerate(self.dataset.targets) if t == cat_index]
            )

    def get_dl(self, batch_size=32, shuffle=True, drop_last=True, ds_type="cat"):
        if ds_type in ["cat", "dog"]:
            data_set_type = self.dog_dataset if ds_type == "dog" else self.cat_dataset
        else:
            data_set_type = self.dataset
        return torch.utils.data.DataLoader(data_set_type, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    def view_training_data(self, ds_type):
        """
        View training data
        Function that will show a batch of training images
        Utalises Matplotlib to show the images and its animation features.
        """
        # Plot some training images
        real_batch = next(iter(self.get_dl(ds_type=ds_type)))

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
        for _ in range(10):
            real_batch = next(iter(self.get_dl(ds_type=ds_type)))
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
        ani.save(f"{self.path}/dynamic_images{ds_type}.mp4")
        plt.show()

    def __call__(self, ds_type, view_train=False):
        if ds_type in ["cat", "dog"]:
            self.set_animal_path()
        else:
            self.set_human_path()
        self.create_data()
        if view_train:
            self.view_training_data(ds_type)


# test = Data()(ds_type="human", view_train=True)
