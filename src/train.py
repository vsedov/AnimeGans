# These are helper functions, if you want them imported in
# from src.core import hp


# These are helper functions, if you want them imported in
# from src.core import hp


import os
import pathlib
from argparse import ArgumentParser

import loguru
import torch
import tqdm
from torch import nn, optim
from torchvision import utils as vutils

import wandb
from src.core import hc
from src.create_data.create_local_dataset import generate_train_loader
from src.models.ACGAN import Discriminator, Generator
from src.utils.torch_utils import *

# These are helper functions, if you want them imported in
# from src.core import hp

DEVICE = torch.device(hc.DEFAULT_DEVICE)
log = loguru.logger

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
        default=5000,
        help="Number of iterations to train Generator",
    )
    parser.add_argument(
        "-G",
        "--extra_generator_layers",
        type=int,
        default=1,
        help="Number of extra layers to train Generator",
    )
    parser.add_argument(
        "-D",
        "--extra_discriminator_layers",
        type=int,
        default=0,
        help="Number of extra layers to train Discriminator",
    )
    parser.add_argument(
        "-C",
        "--check_point_save_split ",
        type=int,
        default=0,
        help="Add a checkpoint split, Number of epochs you want to save your models",
    )

    parser.add_argument(
        "-b", "--batch_size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "-s",
        "--sample_dir",
        type=str,
        default=f"{hc.DIR}results/samples",
        help="Directory to store generated images",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        type=str,
        default=f"{hc.DIR}results/checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--sample", type=int, default=70, help="Sample every n steps"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0002,
        help="Learning rate gen and discriminator",
    )
    # I could modify the parameters : as this is made for acgan
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Momentum term in Adam optimizer",
    )

    # Wandb parameters
    parser.add_argument("--wandb", type=str, default="true", help="Use wandb")

    parser.add_argument(
        "--wandb_project", type=str, default="core", help="Use Project_scope"
    )
    parser.add_argument(
        "--wandb_name", type=str, default="acgan", help="Use project_name"
    )

    parser.add_argument(
        "--overwrite",
        type=str,
        help="Path overwrite, such that if you wish to use this : Batchsize:epoch_ammount is required for given directory 64:120",
    )

    parser.add_argument(
        "-t",
        "--extra_train_model_type",
        type=str,
        help="Use best model instead [number:{}, best]",
    )

    return parser.parse_args()


def save_both(G, D, G_optim, D_optim, checkpoint_dir, epoch, is_best=False):
    suffix = "" if is_best else str(epoch)
    save_model(
        model=G,
        optimizer=G_optim,
        step=epoch,
        file_path=os.path.join(
            checkpoint_dir,
            "G_{}{}.ckpt".format("best_" if is_best else "", suffix),
        ),
    )
    save_model(
        model=D,
        optimizer=D_optim,
        step=epoch,
        file_path=os.path.join(
            checkpoint_dir,
            "D_{}{}.ckpt".format("best_" if is_best else "", suffix),
        ),
    )


def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def initialize_models_and_optimizers(
    args, hair_classes, eye_classes, latent_dim, num_classes
):
    G = Generator(
        latent_dim=latent_dim,
        class_dim=num_classes,
        extra_layers=args.extra_generator_layers,
    ).to(DEVICE)
    D = Discriminator(
        hair_classes=hair_classes,
        eye_classes=eye_classes,
        extra_layers=args.extra_discriminator_layers,
    ).to(DEVICE)

    if args.wandb == "true":
        wandb.watch(G)
        wandb.watch(D)

    G_optim = optim.Adam(G.parameters(), betas=[args.beta, 0.999], lr=args.lr)
    D_optim = optim.Adam(D.parameters(), betas=[args.beta, 0.999], lr=args.lr)

    return G, G_optim, D, D_optim


def load_checkpoint(args, checkpoint_dir, G, G_optim, D, D_optim):
    start_step = 0
    max_n = -1
    if args.extra_train_model_type == "best":
        G, G_optim, D, D_optim, start_step, max_n = load_model(
            G,
            G_optim,
            os.path.join(checkpoint_dir, "G_best_model.ckpt"),
        )
        D, D_optim, start_step, max_n = load_model(
            D,
            D_optim,
            os.path.join(checkpoint_dir, "D_best_model.ckpt"),
        )

    # elif args.extra_train_model_type == "number":
    # Check if it_containsnumber
    elif (x := args.extra_train_model_type) is not None and x.startswith(
        "number"
    ):
        splited_data = args.extra_train_model_type.split(":")
        if len(splited_data) == 1:

            models = list(pathlib.Path(checkpoint_dir).glob("*.ckpt"))
            model_filenames = list(map(lambda x: x.name, models))

            for filename in model_filenames:
                try:
                    step = int(filename.split("_")[-1].split(".")[0])

                    max_n = max(max_n, step)
                    print(max_n)
                except ValueError:
                    pass

            if max_n != -1:
                G, G_optim, start_step = load_model(
                    G,
                    G_optim,
                    os.path.join(checkpoint_dir, "G_{}.ckpt".format(max_n)),
                )
                D, D_optim, start_step = load_model(
                    D,
                    D_optim,
                    os.path.join(checkpoint_dir, "D_{}.ckpt".format(max_n)),
                )

                print("Epoch start: ", start_step)
        else:
            max_n = splited_data[-1]
            G, G_optim, start_step = load_model(
                G,
                G_optim,
                os.path.join(checkpoint_dir, "G_{}.ckpt".format(max_n)),
            )
            D, D_optim, start_step = load_model(
                D,
                D_optim,
                os.path.join(checkpoint_dir, "D_{}.ckpt".format(max_n)),
            )
        print(splited_data)

    return G, G_optim, D, D_optim, start_step


def main(
    args,
):
    # Define configuration batch_size = args.batch_size iterations = args.iterations hair_classes, eye_classes = len(hair), len(eyes) num_classes = hair_classes + eye_classes latent_dim = 128 smooth = 0.9

    if args.wandb == "true":
        wandb.init(project=args.wandb_project, name=args.wandb_name)
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

    # Define configuration
    batch_size = args.batch_size
    iterations = args.iterations
    hair_classes, eye_classes = len(hair), len(eyes)
    num_classes = hair_classes + eye_classes
    latent_dim = 128
    smooth = 0.9

    if args.overwrite:
        x, y = args.overwrite.split(":")
        config = "ACGAN-[{}]-[{}]".format(x, y)
    else:
        config = "ACGAN-[{}]-[{}]".format(batch_size, iterations)

    # Create directories
    random_sample_dir = os.path.join(
        args.sample_dir, config, "random_generation"
    )
    fixed_attribute_dir = os.path.join(
        args.sample_dir, config, "fixed_attributes"
    )
    checkpoint_dir = os.path.join(args.checkpoint_dir, config)

    directories = [random_sample_dir, fixed_attribute_dir, checkpoint_dir]
    create_directories(directories)

    # Initialize models and optimizers
    G, G_optim, D, D_optim = initialize_models_and_optimizers(
        args, hair_classes, eye_classes, latent_dim, num_classes
    )

    # Load checkpoint if it exists
    G, G_optim, D, D_optim, start_step = load_checkpoint(
        args, checkpoint_dir, G, G_optim, D, D_optim
    )

    # Define loss function
    criterion = nn.BCELoss()

    if args.wandb == "true":
        wandb.watch(criterion)

    #  ╭────────────────────────────────────────────────────────────────────╮
    #  │     start Training                                                 │
    #  ╰────────────────────────────────────────────────────────────────────╯

    best_g_loss = float("inf")
    train_loader = generate_train_loader(batch_size=batch_size)
    for epoch in tqdm.trange(iterations, desc="Epoch Loop"):
        if epoch < start_step:
            print(
                f"Skiiping Epoch {epoch} -> {epoch + 1} under iterations : {iterations}"
            )
            continue

        for step_i, (real_img, hair_tags, eye_tags) in enumerate(
            tqdm.tqdm(train_loader, desc="Inner Epoch Loop")
        ):
            real_label = torch.ones(batch_size, device=DEVICE)
            fake_label = torch.zeros(batch_size, device=DEVICE)
            soft_label = torch.Tensor(batch_size).uniform_(smooth, 1).to(DEVICE)
            real_img, hair_tags, eye_tags = (
                real_img.to(DEVICE),
                hair_tags.to(DEVICE),
                eye_tags.to(DEVICE),
            )

            # Train discriminator
            z = torch.randn(batch_size, latent_dim, device=DEVICE)
            fake_tag = get_random_label(
                batch_size=batch_size,
                hair_classes=hair_classes,
                eye_classes=eye_classes,
            ).to(DEVICE)
            fake_img = G(z, fake_tag).to(DEVICE)

            real_score, real_hair_predict, real_eye_predict = D(real_img)
            fake_score, _, _ = D(fake_img)
            # Reshale all values to batch_size now

            for x in [real_score, real_hair_predict, real_eye_predict]:
                log.log(5, x.shape)
            #  ╭────────────────────────────────────────────────────────────────────╮
            #  │                                                                    │
            #  │             THERE IS AN ERROR HERE< ONCE THE HAIR AND EYE AUX      │
            #  │  LOSS DROPS MODEL BLOWS UP                                         │
            #  │                                                                    │
            #  ╰────────────────────────────────────────────────────────────────────╯

            real_discrim_loss = criterion(real_score, soft_label)
            fake_discrim_loss = criterion(fake_score, fake_label)

            #  REVISIT: (vsedov) (16:41:06 - 13/03/23): Fix mode collapse issue
            #  with respect to eye loss
            real_hair_aux_loss = criterion(real_hair_predict, hair_tags)
            real_eye_aux_loss = criterion(real_eye_predict, eye_tags)
            real_classifier_loss = real_hair_aux_loss + real_eye_aux_loss

            discrim_loss = (real_discrim_loss + fake_discrim_loss) * 0.5

            D_loss = discrim_loss + real_classifier_loss
            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()

            if args.wandb == "true":
                wandb.log(
                    {
                        "step": step_i,
                        "real_discrim_loss": real_discrim_loss.item(),
                        "fake_discrim_loss": fake_discrim_loss.item(),
                        "real_hair_aux_loss": real_hair_aux_loss.item(),
                        "real_eye_aux_loss": real_eye_aux_loss.item(),
                        "real_classifier_loss hair + eye": real_classifier_loss.item(),
                    },
                )

            # Train generator
            z = torch.randn(batch_size, latent_dim, device=DEVICE)
            fake_tag = get_random_label(
                batch_size=batch_size,
                hair_classes=hair_classes,
                eye_classes=eye_classes,
            ).to(DEVICE)

            hair_tag = fake_tag[:, 0:hair_classes]
            eye_tag = fake_tag[:, hair_classes:]
            fake_img = G(z, fake_tag).to(DEVICE)

            fake_score, hair_predict, eye_predict = D(fake_img)
            discrim_loss = criterion(fake_score, real_label)
            hair_aux_loss = criterion(hair_predict, hair_tag)
            eye_aux_loss = criterion(eye_predict, eye_tag)
            classifier_loss = hair_aux_loss + eye_aux_loss

            G_loss = classifier_loss + discrim_loss
            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()

            # Print the data
            print(
                f"Epoch: {epoch + 1}/{iterations} Iteration: {step_i + 1}/{len(train_loader)} Loss D: {D_loss.item():.4f} Loss G: {G_loss.item():.4f}"
            )
            print("\n")

            if G_loss.item() < best_g_loss:
                best_g_loss = G_loss.item()

                print(f"Saving Best Model {G_loss.item()} best loss ")
                save_both(
                    G, D, G_optim, D_optim, checkpoint_dir, epoch, is_best=True
                )

            if args.wandb == "true":
                wandb.log(
                    {
                        "D_loss": D_loss.item(),
                        "G_loss": G_loss.item(),
                        "discrim_loss": discrim_loss,
                        "gen_loss / Classifier loss ": classifier_loss,
                        "hair_aux_loss": hair_aux_loss,
                        "eye_aux_loss": eye_aux_loss,
                    }
                )

            if epoch == 0 and step_i == 0:
                vutils.save_image(
                    real_img, os.path.join(random_sample_dir, "real.png")
                )

            if step_i % args.sample == 0:
                vutils.save_image(
                    fake_img.data.view(batch_size, 3, 64, 64),
                    os.path.join(
                        random_sample_dir,
                        "fake_step_{}_{}.png".format(epoch, step_i),
                    ),
                )

            # if (step_i == 0 and args.check_point_save_split == 0) or (
            #     args.check_point_save_split != 0
            #     and epoch % args.check_point_save_split == 0
            #     and step_i == 0
            # ):
            if epoch % 100 == 0 and step_i == 0:
                save_both(G, D, G_optim, D_optim, checkpoint_dir, epoch)

            generate_by_attributes(
                model=G,
                device=DEVICE,
                step=epoch,
                latent_dim=latent_dim,
                hair_classes=hair_classes,
                eye_classes=eye_classes,
                sample_dir=fixed_attribute_dir,
            )

    if args.wandb == "true":
        wandb.finish()


def run():
    main(parse_args())


if __name__ == "__main__":
    run()
