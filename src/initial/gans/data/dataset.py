import os
import zipfile

from src.core import hc
from src.data.getdata import DSDownloader


def set_up_dog_ds():
    """
    Setup Cat and dog ds:
    We split the ds, into two paths and extract any extended zip files required.
    This is done automatically.

    Zip path: is the core directory of where the data should go.

    Unzip file: Animal ds path extraction of the initial ds
    command: Command to call from kaggle
    name: the extracted zip file that we will get.
    """
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


def set_up_face_ds():
    """
    Setup Face Dataset:

    No external extraction will be required.

    Quite a large dataset.
    """
    zip_path = f"{hc.DIR}initial/gans/data"
    unzip_path = f"{hc.DIR}/initial/gans/data/human_faces/"
    command = f"kaggle datasets download -d jessicali9530/celeba-dataset -p {zip_path} --force"
    name = "celeba-dataset.zip"
    DSDownloader(command, zip_path, unzip_path, name)()
