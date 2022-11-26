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
        "initial": {
            "gan": (valid := (True)),
            "conv": not valid,
            "params": {
                "latent_vector": 100,
                "image_size": 64,
                "noise_vector": 128,  #  REVISIT: (vsedov) (04:27:45 - 26/11/22): Need to double check if this is required or not
                "output": 3,
                "epoch_amount": [2, 4, 30, 100, 300, 500, 1000][-1],  # Train on 1k epochs overnight
                "batch_size": [32, 64, 128, 256, 512, 1024][-3],  # Generalisation issue, but lets see
                "learning_rate": [0.0001, 0.0005, 0.001, 0.00146][0],
            },
        },
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
