import os

from src.core import hc

zip_path = f"{hc.DIR}/data/zip/"
command = ""  # Kaggle command


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
    Boolean
        True if the file exists else False
    """
    return True if str(os.path.exists(filename)) else False


def check_zip():
    """Check if zip file exists and if not, download it using main kggle command"""
    if check_current_path_file(f"{zip_path}heartbeat.zip"):
        os.system(command)


def force_replace_zip():
    """Force replace zip for what ever reason"""
    if check_folder(zip_path):
        for files in os.listdir(zip_path):
            os.remove(files)
    check_zip()
