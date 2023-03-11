import glob
import os
import re
from argparse import ArgumentParser

import torch
import tqdm
from torch import nn, optim
from torchvision import utils as vutils

import wandb
from src.create_data.create_local_dataset import train_loader
from src.models.ACGAN import Discriminator, Generator
from src.utils.torch_utils import *
from src.utils.utils import current_path

current_path = current_path()

hair = [
    "orange",
    "white",
    "aqua",
    "gray",
    "green",
    "red",
    "purple",
    "pink",
    "blue",
    "black",
    "brown",
    "blonde",
]
eyes = [
    "gray",
    "black",
    "orange",
    "pink",
    "yellow",
    "aqua",
    "purple",
    "green",
    "brown",
    "red",
    "blue",
]


def parse_args():
    """Parses the command line arguments and returns them."""
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations to train ACGAN",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "-s",
        "--sample_dir",
        type=str,
        default=f"{current_path}/results/samples",
        help="Directory to store generated images",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        type=str,
        default=f"{current_path}/results/checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--sample", type=int, default=70, help="Sample every n steps"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate of ACGAN"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Momentum term in Adam optimizer",
    )
    parser.add_argument("--wandb", type=str, default="true", help="Use wandb")
    parser.add_argument(
        "--wandb_project", type=str, default="core", help="Use wandb"
    )

    return parser.parse_args()

def main(args):
    if args.wandb == "true":
        wandb.init(project=args.wandb_project, entity="core")
        wandb.config.update(
            {
                "batch_size": args.batch_size,
                "iterations": args.iterations,
                "lr": args.lr,
                "beta": args.beta,
                "sample_dir": args.sample_dir,
                "checkpoint_dir": args.checkpoint_dir,
                "sample": args.sample,
                "hair_classes": len(hair),
                "eye_classes": len(eyes),
            }
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define configuration
    batch_size = args.batch_size
    iterations = args.iterations
    hair_classes, eye_classes = len(hair), len(eyes)
    num_classes = hair_classes + eye_classes
    latent_dim = 128
    smooth = 0.9
    config = "ACGAN-[{}]-[{}]".format(batch_size, iterations)

