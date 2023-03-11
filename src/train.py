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

    # Create directories
    random_sample_dir = os.path.join(
        args.sample_dir, config, "random_generation"
    )
    fixed_attribute_dir = os.path.join(
        args.sample_dir, config, "fixed_attributes"
    )
    checkpoint_dir = os.path.join(args.checkpoint_dir, config)
    for directory in [random_sample_dir, fixed_attribute_dir, checkpoint_dir]:
        os.makedirs(directory, exist_ok=True)

    # Initialize models and optimizers
    G = Generator(latent_dim=latent_dim, class_dim=num_classes).to(device)
    D = Discriminator(hair_classes=hair_classes, eye_classes=eye_classes).to(
        device
    )
    if args.wandb == "true":
        wandb.watch(G)
        wandb.watch(D)

    G_optim = optim.Adam(G.parameters(), betas=[args.beta, 0.999], lr=args.lr)
    D_optim = optim.Adam(D.parameters(), betas=[args.beta, 0.999], lr=args.lr)

    # Load checkpoint if it exists
    start_step = 0
    models = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    max_n = -1
    for model in models:
        n = int(re.findall(r"\d+", model)[-1])
        max_n = max(max_n, n)
    if max_n != -1:
        G, G_optim, start_step = load_model(
            G, G_optim, os.path.join(checkpoint_dir, "G_{}.ckpt".format(max_n))
        )
        D, D_optim, start_step = load_model(
            D, D_optim, os.path.join(checkpoint_dir, "D_{}.ckpt".format(max_n))
        )
        print("Epoch start: ", start_step)

    # Define loss function
    criterion = nn.BCELoss()

    if args.wandb == "true":
        wandb.watch(criterion)
