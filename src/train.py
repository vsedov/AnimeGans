#  ╭────────────────────────────────────────────────────────────────────╮
#  │ gradient penalty technique to improve the stability of the         │
#  │  GAN training.                                                     │
#  │ Specifically, it adds a penalty term to the discriminator's        │
#  │  loss, which                                                       │
#  │ encourages the gradients of the discriminator to be close to 1 for │
#  │ all points in the input space This helps prevent the discriminator │
#  │ from being too powerful, which can lead to mode collapse and       │
#  │  other issues.                                                     │
#  │                                                                    │
#  │ The code computes the gradient penalty by interpolating            │
#  │  between real                                                      │
#  │ and fake images, and then computing the norm of the gradients      │
#  │ of the discriminator's output with respect to the interpolated     │
#  │ images. The penalty is then added to the discriminator's loss,     │
#  │ along with the usual adversarial loss and auxiliary losses for     │
#  │ the hair and eye tags.                                             │
#  │                                                                    │
#  │ The code also trains the generator as usual, with a loss           │
#  │ that includes the adversarial loss and auxiliary losses for        │
#  │ the hair and eye tags. If wandb logging is enabled, it logs        │
#  │ various loss values for both the discriminator and generator.      │
#  │                                                                    │
#  ╰────────────────────────────────────────────────────────────────────╯

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
    # "orange",
    "white",  #
    # "aqua",
    # "gray",  #
    "green",
    "red",
    # "purple",
    "pink",  #
    "blue",
    "black",
    "brown",  #
    "blonde",
]
eyes = [
    # "gray",
    # "black",
    # "orange",
    "pink",
    "yellow",
    "aqua",
    "purple",
    "green",
    "brown",
    "red",
    "blue",
]

print(len(eyes), len(hair))


def parse_args():
    """This function parses the command line arguments and returns them as an object."""
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
        "--cp_per_save",
        type=int,
        default=10,  # have it save every 50 epochs
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

    parser.add_argument(
        "-L",
        "--lambda_gp",
        type=float,
        default=0.1,
        help="Gradient penalty lambda",
    )

    return parser.parse_args()


def save_both(G, D, G_optim, D_optim, checkpoint_dir, epoch, is_best=False):
    """
    This function saves the Generator and Discriminator
    models along with their respective optimizers to the given
    checkpoint directory.
    """
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
    # Define configuration batch_size = args.batch_size
    # iterations = args.iterations
    # hair_classes, eye_classes = len(hair),
    # len(eyes) num_classes = hair_classes + eye_classes
    # latent_dim = 128
    # smooth = 0.9
    if args.wandb == "true":
        wandb.init(project=args.wandb_project, name=args.wandb_name)
        config = vars(args)
        config.update(
            {
                "len_hair_classes": len(hair),
                "len_eye_classes": len(eyes),
                "hair_classes": hair,
                "eye_classes": eyes,
                "smooth": 0.9,
                "latent_dim": 128,
            }
        )
        wandb.config.update(config)

    # Define configuration
    batch_size = args.batch_size
    iterations = args.iterations
    hair_classes, eye_classes = len(hair), len(eyes)
    num_classes = hair_classes + eye_classes
    print(hair_classes, eye_classes)
    print("Num Classes total, ", num_classes)

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
    # Wrong output this does
    # criterion = nn.BCEWithLogitsLoss()

    if args.wandb == "true":
        wandb.watch(criterion)

    #  ╭────────────────────────────────────────────────────────────────────╮
    #  │     start Training                                                 │
    #  ╰────────────────────────────────────────────────────────────────────╯
    # hair_len=hair_classes, eye_len=eye_classes,
    train_loader = generate_train_loader(
        hair_classes=hair_classes,
        eye_classes=eye_classes,
        batch_size=batch_size,
    )
    if args.lambda_gp != 0:
        lambda_gp = args.lambda_gp

    best_g_loss = float("inf")
    for epoch in tqdm.trange(iterations, desc="Epoch Loop"):
        if epoch < start_step:
            tqdm.tqdm.write(
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
                use_numpy=True,
            ).to(DEVICE)

            fake_img = G(z, fake_tag).to(DEVICE)

            real_score, real_hair_predict, real_eye_predict = D(real_img)
            fake_score, _, _ = D(fake_img)

            # Compute gradient penalty
            if lambda_gp != 0:
                alpha = torch.rand(batch_size, 1, 1, 1).to(DEVICE)
                interpolates = (
                    alpha * real_img + (1 - alpha) * fake_img
                ).requires_grad_(True)
                d_interpolates, _, _ = D(interpolates)
                gradients = torch.autograd.grad(
                    outputs=d_interpolates,
                    inputs=interpolates,
                    grad_outputs=torch.ones_like(d_interpolates),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                gradient_penalty = (
                    (gradients.norm(2, dim=1) - 1) ** 2
                ).mean() * lambda_gp
            else:
                gradient_penalty = 0

            # Compute losses
            real_discrim_loss = criterion(real_score, soft_label)
            fake_discrim_loss = criterion(fake_score, fake_label)
            real_hair_aux_loss = criterion(real_hair_predict, hair_tags)
            real_eye_aux_loss = criterion(real_eye_predict, eye_tags)
            real_classifier_loss = real_hair_aux_loss + real_eye_aux_loss

            discrim_loss = (real_discrim_loss + fake_discrim_loss) * 0.5
            D_loss = discrim_loss + real_classifier_loss + gradient_penalty

            # Update discriminator
            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()

            # Train generator
            z = torch.randn(batch_size, latent_dim, device=DEVICE)
            fake_tag = get_random_label(
                batch_size=batch_size,
                hair_classes=hair_classes,
                eye_classes=eye_classes,
                use_numpy=False,
            ).to(DEVICE)
            fake_img = G(z, fake_tag).to(DEVICE)

            fake_score, fake_hair_predict, fake_eye_predict = D(fake_img)
            fake_discrim_loss = criterion(fake_score, real_label)
            fake_hair_aux_loss = criterion(
                fake_hair_predict, fake_tag[:, :hair_classes]
            )
            fake_eye_aux_loss = criterion(
                fake_eye_predict, fake_tag[:, hair_classes:]
            )
            fake_classifier_loss = fake_hair_aux_loss + fake_eye_aux_loss

            G_loss = fake_discrim_loss + fake_classifier_loss
            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()

            tqdm.tqdm.write(
                f"Epoch: {epoch + 1}/{iterations} Iteration: {step_i + 1}/{len(train_loader)} Loss D: {D_loss.item():.4f} Loss G: {G_loss.item():.4f}"
            )

            if G_loss.item() < best_g_loss:
                best_g_loss = G_loss.item()

                tqdm.tqdm.write(f"Saving Best Model {G_loss.item()} best loss ")
                save_both(
                    G, D, G_optim, D_optim, checkpoint_dir, epoch, is_best=True
                )

            if args.wandb == "true":

                wandb.log(
                    {
                        "step": step_i,
                        "D_loss": D_loss.item(),
                        "G_loss": G_loss.item(),
                        "real_discrim_loss": real_discrim_loss.item(),
                        "fake_discrim_loss": fake_discrim_loss.item(),
                        "real_hair_aux_loss": real_hair_aux_loss.item(),
                        "real_eye_aux_loss": real_eye_aux_loss.item(),
                        "fake_hair_aux_loss": fake_hair_aux_loss.item(),
                        "fake_eye_aux_loss": fake_eye_aux_loss.item(),
                        "real_classifier_loss hair + eye": real_classifier_loss.item(),
                        "fake_classifier_loss hair + eye": fake_classifier_loss.item(),
                        "gradient_penalty": gradient_penalty.item(),
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

            if (step_i == 0 and args.cp_per_save == 0) or (
                args.cp_per_save != 0
                and epoch % args.cp_per_save == 0
                and step_i == 0
            ):
                # if epoch % 100 == 0 and step_i == 0:
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
