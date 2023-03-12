import os
from argparse import ArgumentParser
from collections import namedtuple

import torch

from src.models.ACGAN import Generator
from src.utils.torch_utils import *
from src.utils.utils import current_path

current_path = current_path()


def parse_args():
    Args = namedtuple(
        "Args",
        [
            "type",
            "hair",
            "eye",
            "sample_dir",
            "batch_size",
            "epoch",
            "check_point_number",
            "gen_model_dir",
        ],
    )
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--type",
        help="Type of anime generation.",
        choices=[
            "fix_noise",
            "fix_hair_eye",
            "change_hair",
            "change_eye",
            "interpolate",
        ],
        default="fix_noise",
        type=str,
    )
    parser.add_argument(
        "--hair",
        help="Determine the hair color of the anime characters.",
        default=hair_mapping[2],
        choices=hair_mapping,
        type=str,
    )
    parser.add_argument(
        "--eye",
        help="Determine the eye color of the anime characters.",
        default=eye_mapping[2],
        choices=eye_mapping,
        type=str,
    )
    parser.add_argument(
        "-s",
        "--sample_dir",
        help="Folder to save the generated samples.",
        default=f"{current_path}/results/generated",
        type=str,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Batch size used during training.",
        default=64,
        type=int,
    )
    parser.add_argument(
        "-e",
        "--epoch",
        help="Number of epochs used during training.",
        default=5,
        type=int,
    )
    parser.add_argument(
        "-c",
        "--check_point_number",
        help="Checkpoint number to use.",
        default=4,
        type=int,
    )

    args = parser.parse_args()
    args.gen_model_dir = f"{current_path}/results/checkpoints/ACGAN-[{args.batch_size}]-[{args.epoch}]/G_{args.check_point_number}.ckpt"
    return Args(*args.__dict__.values())


def main(args):
    """
    Main function.

    Args:
        args: Arguments.

    Notes:
        - The function will generate anime characters based on the given arguments.
        Generated samples will be saved in the folder specified by the argument `sample_dir`.
        - The function will generate anime characters based on the given arguments.

    Returns:
        None

    """
    os.makedirs(args.sample_dir, exist_ok=True)
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    latent_dim = 128
    hair_classes = len(hair_mapping)
    eye_classes = len(eye_mapping)

    G = Generator(latent_dim, hair_classes + eye_classes).to(device)
    prev_state = torch.load(args.gen_model_dir)
    G.load_state_dict(prev_state["model"])
    G = G.eval()

    action_map = {
        "fix_hair_eye": lambda: generate_by_attributes(
            G,
            device,
            latent_dim,
            hair_classes,
            eye_classes,
            args.sample_dir,
            hair_color=args.hair,
            eye_color=args.eye,
        ),
        "change_eye": lambda: eye_grad(
            G, device, latent_dim, hair_classes, eye_classes, args.sample_dir
        ),
        "change_hair": lambda: hair_grad(
            G, device, latent_dim, hair_classes, eye_classes, args.sample_dir
        ),
        "interpolate": lambda: interpolate(
            G, device, latent_dim, hair_classes, eye_classes, args.sample_dir
        ),
        "fix_noise": lambda: fix_noise(
            G, device, latent_dim, hair_classes, eye_classes, args.sample_dir
        ),
    }

    action_map.get(args.type, lambda: None)()


def run():
    parse = parse_args()
    print("Help")
    data = parse
    for key in data._fields:
        print(key)
    print("Done")
    main(parse)
