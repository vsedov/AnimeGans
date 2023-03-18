import os
from argparse import ArgumentParser
from collections import namedtuple

import torch

from src.core import hc
from src.models.ACGAN import Generator
from src.utils.torch_utils import *


def generate_images(
    args, G, device, latent_dim, hair_classes, eye_classes, output_dir=None
):
    action_map = {
        "fix_hair_eye": lambda: generate_by_attributes(
            G,
            device,
            latent_dim,
            hair_classes,
            eye_classes,
            args.sample_dir if output_dir is None else output_dir,
            hair_color=args.hair,
            eye_color=args.eye,
        ),
        "change_eye": lambda: eye_grad(
            G,
            device,
            latent_dim,
            hair_classes,
            eye_classes,
            args.sample_dir if output_dir is None else output_dir,
        ),
        "change_hair": lambda: hair_grad(
            G,
            device,
            latent_dim,
            hair_classes,
            eye_classes,
            args.sample_dir if output_dir is None else output_dir,
        ),
        "interpolate": lambda: interpolate(
            G,
            device,
            latent_dim,
            hair_classes,
            eye_classes,
            args.sample_dir if output_dir is None else output_dir,
        ),
        "fix_noise": lambda: fix_noise(
            G,
            device,
            latent_dim,
            hair_classes,
            eye_classes,
            args.sample_dir if output_dir is None else output_dir,
        ),
    }

    action_map.get(args.type, lambda: None)()


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
            "extra_generator_layers",
            "range",
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
        default=f"{hc.DIR}results/generated",
        type=str,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Batch size used during training (Why) : Because you May retrain your dataset on a previous model, if you do something where you train on epoch 100, and then you retrain on that data on epoch 200, in this instance you would have to refer to  its respected batch",
        default=64,
        type=int,
    )
    parser.add_argument(
        "-e",
        "--epoch",
        help="Number of epochs used during training., if previous models are used, please refer to a given batch number that you would default to train with. ",
        default=100,
        type=int,
    )
    parser.add_argument(
        "-c",
        "--check_point_number",
        help="Checkpoint number to use or you can use 'best'",
        default=100,
        # type=Union[int, str],
    )

    parser.add_argument(
        "-E",
        "--extra_generator_layers",
        help="Add extra layers to the generator.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-r",
        "--range",
        help="Range of checkpoint numbers to use, separated by a colon (e.g. 10:50:STEP).",
        type=str,
    )

    args = parser.parse_args()
    if args.check_point_number == "best":
        args.gen_model_dir = f"{hc.DIR}results/checkpoints/ACGAN-[{args.batch_size}]-[{args.epoch}]/G_best_.ckpt"
    else:
        args.gen_model_dir = f"{hc.DIR}results/checkpoints/ACGAN-[{args.batch_size}]-[{args.epoch}]/G_{args.check_point_number}.ckpt"

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
    device = torch.device(hc.DEFAULT_DEVICE)
    latent_dim = 128
    hair_classes = len(hair_mapping)
    eye_classes = len(eye_mapping)

    print(args.extra_generator_layers)
    print(type(args.extra_generator_layers))

    G = Generator(
        latent_dim,
        hair_classes + eye_classes,
        extra_layers=args.extra_generator_layers,
    ).to(device)
    if args.range:
        start, end, step = map(int, str(args.range).split(":"))
        if args.check_point_number == "best":
            return ValueError("Cannot use `best` with `range`.")

        for i in range(start, end + 1, step):
            gen_model_dir = f"{hc.DIR}results/checkpoints/ACGAN-[{args.batch_size}]-[{args.epoch}]/G_{i}.ckpt"
            prev_state = torch.load(gen_model_dir)
            G.load_state_dict(prev_state["model"])
            G = G.eval()
            output_dir = os.path.join(args.sample_dir, "batch_images", f"G_{i}")
            os.makedirs(output_dir, exist_ok=True)
            generate_images(
                args,
                G,
                device,
                latent_dim,
                hair_classes,
                eye_classes,
                output_dir,
            )
    else:

        prev_state = torch.load(args.gen_model_dir)
        G.load_state_dict(prev_state["model"])
        G = G.eval()
        generate_images(args, G, device, latent_dim, hair_classes, eye_classes)

    print("Completed generating images for the given arguments:")
    print("\n")
    print(args)
    print("\n")
    print(
        f"Please refer to folder : {args.sample_dir} to see the generated images."
    )


def run():
    parse = parse_args()
    main(parse)


if __name__ == "__main__":
    run()
