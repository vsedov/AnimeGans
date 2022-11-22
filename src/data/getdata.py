import os
import zipfile

from heart.core import hc
from heart.log import get_logger

log = get_logger(__name__)

zip_path = f"{hc.DIR}/data/zip/"
command = f"kaggle datasets download -d name/path -p {zip_path}"


def check_folder(folder):
    """Check folder

    Parameters
    ----------
    folder : Directory path to a folder

    Returns
    -------
    BOolean
        True if it is a folder and that the folder contains some item
    """
    return True if os.path.isdir(folder) and os.listdir(folder) else False


def check_current_path_file(filename):
    """Check current path file

    Parameters
    ----------
    filename : File name / needs full path - use constants for this
        Full path to x.py file

    Returns
    -------
    Booelan
        True if the file exists else False
    """
    return True if str(os.path.exists(filename)) else False


def check_zip():
    """Check if zip file exists and if not, download it using main kggle command"""
    if check_current_path_file(f"{zip_path}anime.zip"):
        log.info("Downloading zip file")
        os.system(command)


def force_replace_zip():
    """Force replace zip for what ever reason"""
    if check_folder(zip_path):
        for files in os.listdir(zip_path):
            os.remove(files)
    check_zip()


def download_dir():
    """
    Download Zip and Daataset
    """
    if check_folder(f"{hc.DIR}/data/anime"):
        return ...

    check_zip()

    with zipfile.ZipFile(f"{zip_path}anime.zip", "r") as zip_ref:
        zip_ref.extractall(f"{hc.DIR}/data/anime")
