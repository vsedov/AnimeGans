import os

from src.data.getdata import DSDownloader

current_path = os.path.dirname(os.path.realpath(__file__))


# NOTE: (vsedov) (00:47:23 - 03/01/23): This File must be only downloaded once
#  TODO: (vsedov) (00:47:39 - 03/01/23): Create a boolean lock or something for
#  this to be valid later down the line.
def get_tags_from_danbooru() -> None:
    """Fetch Tags from tags.txt"""
    with open(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "tags.txt")
    ) as f:
        tags = f.read()

    for tag in tags.split("\n"):
        print(tag)
        os.system(
            f'gallery-dl --range 1-100 "https://danbooru.donmai.us/posts?tags={tag}"'
        )


def kaggle_data_set() -> None:
    ds = {
        "splcher/animefacedataset",
        "soumikrakshit/anime-faces",
        "scribbless/another-anime-face-dataset",
    }
    for i in ds:
        command = f"kaggle datasets download -d {i}"
        zip_path = os.path.join(os.path.dirname(__file__), "data/")
        unzip_path = os.path.join(os.path.dirname(zip_path))
        zip_name = ""
        # DSDownloader(
        #     command,
        # )
        print(command)
        print(zip_path)
        print(unzip_path)
        print(zip_name)


kaggle_data_set()
