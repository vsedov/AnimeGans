import os
from subprocess import CalledProcessError, check_output

import torch
from dotenv import load_dotenv

load_dotenv()


def root():
    """returns the absolute path of the repository root"""
    try:
        base = check_output("git rev-parse --show-toplevel", shell=True)
    except CalledProcessError:
        raise IOError("Current working directory is not a git repository")
    return base.decode("utf-8").strip()


def constants():
    return {
        "DEFAULT_DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "KAGGLE_USERNAME": os.getenv("KAGGLE_USERNAME"),
        "KAGGLE_KEY": os.getenv("KAGGLE_KEY"),
        "DIR": f"{root()}/src/",
        "show_data": True,
        "initial": {"gan": (valid := (True)), "conv": not valid},
    }


def constants_extra():
    """Constants extra : Builtin functions are used
    to call constants: if we have to recall or reuse values we can use this function
    to parse certain constants, which would allow us to evaluate data faster

    Returns
    -------
    Dictionary
        Dictionary of external referred constants
    """
    return {
        "DATASET_DIR": os.path.join(hc.DIR, "data/anime/"),
    }
