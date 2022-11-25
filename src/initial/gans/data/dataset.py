import os
import zipfile

from src.core import hc
from src.data.getdata import DSDownloader


def set_up_dog_ds():
    zip_path = f"{hc.DIR}initial/gans/data"
    unzip_path = f"{hc.DIR}initial/gans/data/animal"
    command = f"kaggle competitions download -c dogs-vs-cats -p {zip_path} --force"
    name = "dogs-vs-cats.zip"
    DSDownloader(command, zip_path, unzip_path, name)()

    data = ["test1.zip", "train.zip"]
    for val in data:
        if os.path.isdir(f"{unzip_path}/{val.removesuffix('.zip')}"):
            # if atleast one exist , we can continue and skip this part
            return

        with zipfile.ZipFile(f"{unzip_path}/{val}") as unzip:
            unzip.extractall(unzip_path)
