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
    """Constants:

    Return a dictionary that will get parsed into
    src.core.HC
    Once parsed on __init__.py : you will have access to global modifiers
    through out the entire process.
    """
    return {
        "DEFAULT_DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "KAGGLE_USERNAME": os.getenv("KAGGLE_USERNAME"),
        "KAGGLE_KEY": os.getenv("KAGGLE_KEY"),
        "DIR": f"{root()}/src/",
        "show_data": True,  # this is for initial, and is a param to show data in a mp4 file
        #  ╭────────────────────────────────────────────────────────────────────╮
        #  │         Core Data that we will be using                            │
        #  ╰────────────────────────────────────────────────────────────────────╯
        "core": {
            "batch_size": [64, 128, 512][1],
            "image_size": 128,
            "nz": 100,
            "ngf": 64,
            "ndf": 64,
            # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
            "epoch": [1 << i for i in range(14)][
                -1
            ],  # List or range of power of 2
            "lr": {
                "stable": 2e-4,
                "momentum_based": 2e-4,  # Need to figure out how to really finish this, or code this thing up.
            },
            "beta_1": 0.5,
            "label_smoothing": 0.1,  # Taken from the gan improvement paper
            "extra_layers_d": 0,
            "extra_layers_g": 1,  # This will allow the generaetor to be slightly more powerful.
        }
        # "initial": {
        #     "train": (run_test := (False)),  # if True, we will not test the model
        #     "test": not run_test,
        #     "gan": (valid := (True)),  # If this is True, then Conv will run the program to show how conv works
        #     # Once again this is more for training, and to understand certain principles. and not for the core project
        #     # the initial folder is based for the first part of the project, purely to understand things.
        #     "conv": not valid,
        #     "params": {
        #         "latent_vector": 100,  # 100 | 128 | 256
        #         "image_size": 64,
        #         "noise_vector": 64,  # This is the amount of noise vectors we want Wise to keep this the same as the image size
        #         "output": 3,
        #         "batch_size": [32, 64, 128, 256, 512, 1024][-2],
        #         "lr": [0.0001, 0.0005, 0.001, 0.00146][0],
        #         "ds_type": "human",  # Human , cat, dog : Human ds May require extra data
        #     },
        # },
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
