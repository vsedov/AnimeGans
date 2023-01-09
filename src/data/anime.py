import os
from multiprocessing import Pool

import animeface
import numpy as np
import torch
import torchvision.datasets as ds
import torchvision.transforms as transforms
from PIL import Image
from icecream import ic

from src.core import hp


class DSBuilder:
    """
    class CLS to construct the folders/organised faces
    Of each tag / label that we want to combine;
    Tags:
    Rem Rezero
    Cute Anime
    Scared

    This will take images from danbooru and create folders/ which will be the baseline for the tags.
    """

    def construct_face_dataset(self):
        dirs = [f for f in os.listdir(self.danbooru_path)]
        imgs = []
        for i in range(len(dirs)):
            sub_dir = os.path.join(self.danbooru_path, dirs[i])
            imgs = imgs + ([os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if f.endswith((".jpg", ".png"))])
        ic(f"There is : {len(dirs)} classes and a total of {len(imgs)}")
        pool = Pool(12)  # 1t workers
        pool.map(self.proc_image, imgs)

    def proc_image(self, img):
        parts = img.split("/")
        label, image_name = parts[-2], parts[-1]
        new_dir_path = os.path.join("gallery-dl", "custom_faces", label)

        try:
            os.makedirs(new_dir_path, exist_ok=True)
        except FileExistsError:
            pass
        new_img_path = os.path.join(new_dir_path, image_name)
        if os.path.exists(new_img_path):
            return
        im = self.load_image(img)
        if im == 0:
            return
        im = im.convert("RGB")
        im.save(new_img_path, "JPEG")

    def load_image(self, img_path):

        im = Image.open(img_path)
        faces = animeface.detect(im)
        prob_list = []
        len_f = len(faces)
        if len_f == 0:
            return 0
        for i in range(len_f):
            prob_list.append(faces[i].likelihood)
        prob_array = np.array(prob_list)
        idx = np.argmax(prob_array)
        face_pos = faces[idx].face.pos

        return self.crop(self.crop_face(im, face_pos, 0.5), 96)

    def crop_face(self, im, face_pos, m):
        x, y, w, h = face_pos.x, face_pos.y, face_pos.width, face_pos.height
        size_x, size_y = im.size
        new_x = max(0, x - m*w)
        new_y = max(0, y - m*h)
        new_w = min(w + 2*m*w, size_x - new_x)
        new_h = min(h + 2*m*h, size_y - new_y)
        return im.crop((new_x, new_y, new_x + new_w, new_y + new_h))

    def crop(self, img, min_side):
        size_x, size_y = img.size
        if size_x > size_y:
            new_width = min_side * size_x / size_y
            new_height = min_side
        else:
            new_width = min_side
            new_height = size_y * min_side / size_x
        im = img.resize((int(new_width), int(new_height)), Image.Resampling.LANCZOS)
        return im.crop((0, 0, min_side, min_side))


class AnimeManualBuilder(DSBuilder):

    def __init__(self, path: str, download_default: bool = False, create_dataset: bool = False,):
        super().__init__()
        self.path = path

        self.create_dataset = create_dataset
        self.use_default = download_default
        self.gallery_path = f"{self.path}/gallery-dl"
        self.danbooru_path = f"{self.gallery_path}/danbooru"

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "tags.txt")) as f:
            self.tags = f.read().split("\n")
        os.chdir(self.path)

    def dir_check(self):
        return os.path.isdir(f"{self.path}/gallery-dl")

    def get_path(self):
        if self.use_default:
            return f"{self.gallery_path}/anime-faces/"
        return f"{self.gallery_path}/custom_faces/"

    def get_data_from_danbooru(self) -> None:
        """Download data from danbooru"""
        if self.dir_check():
            ic("Path exists")

        for tag in self.tags:
            ic(tag)
            if not os.path.isdir(f"{self.danbooru_path}/{tag}"):
                os.system("gallery-dl --range 1-100 " + f" 'https://danbooru.donmai.us/posts?tags={tag}'")

    def get_data_from_default(self) -> None:
        if not self.dir_check() or not os.path.isfile(f"{self.path}/gallery-dl/anime-face.tar.gz"):
            return
        if (not os.path.isdir(f"{self.path}/gallery-dl/anime-face") and self.use_default):
            os.system("tar xzvf anime-faces.tar.gz")

    def __call__(self):
        if self.create_dataset:
            if self.use_default:
                self.get_data_from_default()
            else:
                self.get_data_from_danbooru()
                self.construct_face_dataset()


def create_image_folder(create_dataset: bool = False, use_default: bool = False):
    """Create image folder

    Parameters
    ----------
    create_dataset : bool
        This is a boolean value that states if we want to create the datasets and the file locations meaning that do you
        want to create the images, this by default is set to False. Set this to True, if you are going to run this for
        the first time.
    use_default : bool
        There are two types of datasets with unique tags that you can use right now.
        1. Default dataset which is the big zip file: this is not custom
        2. Custom dataset which will be based on the user. Ideally this dataset will contain certain tags -
            you will have to train this on your own.

    Returns
    -------
        torchvision.datasets.ImageFolder
            ImageFolder of what ever desired tags that you wish to use.
    """
    current_path = os.path.dirname(os.path.realpath(__file__))
    data = AnimeManualBuilder(current_path, download_default=use_default, create_dataset=create_dataset,)
    data()
    return ds.ImageFolder(
        root=data.get_path(),
        transform=transforms.Compose([
            transforms.Resize(hp.get_core("image_size")),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    )


def create_data_loader(dataset, batch_size, shuffle, num_workers):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
