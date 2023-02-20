import argparse
import os
import random
import time

import torch
from icecream import ic
from loguru import logger as log

import wandb
from src.core import hc, hp
from src.data.anime import create_data_loader, create_image_folder
from src.models.dcgan_version_two import DCGanVariantTwoGenerator
from src.utils.wandb import init_wandb

active_logger = False


def load_wandb(project_name, run_name, config):
    init_wandb(project_name, run_name, config)


def check_cuda():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ic(f"Using {device}")
    return device


def set_seeds(opt):
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataRoot", required=True, help="path to dataset")
    parser.add_argument(
        "--workers", type=int, default=2, help="number of data loading workers"
    )
    parser.add_argument(
        "--batchSize", type=int, default=64, help="input batch size"
    )
    parser.add_argument(
        "--imageSize",
        type=int,
        default=64,
        help="the height / width of the input image to network",
    )
    parser.add_argument(
        "--nz", type=int, default=100, help="size of the latent z vector"
    )
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument(
        "--niter", type=int, default=25, help="number of epochs to train for"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="learning rate, default=0.0002"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for adam. default=0.5"
    )
    parser.add_argument("--cuda", action="store_true", help="enables cuda")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="number of GPUs to use"
    )
    parser.add_argument(
        "--netG", default="", help="path to netG (to continue training)"
    )
    parser.add_argument(
        "--netD", default="", help="path to netD (to continue training)"
    )
    parser.add_argument(
        "--outDir",
        default=".",
        help="folder to output images and model checkpoints",
    )
    parser.add_argument(
        "--model",
        type=int,
        default=1,
        help="1 for dcgan, 2 for illustrationGAN-like-GAN",
    )
    parser.add_argument(
        "--d_labelSmooth",
        type=float,
        default=0,
        help='for D, use soft label "1-labelSmooth" for real samples',
    )
    parser.add_argument(
        "--n_extra_layers_d",
        type=int,
        default=0,
        help="number of extra conv layers in D",
    )
    parser.add_argument(
        "--n_extra_layers_g",
        type=int,
        default=1,
        help="number of extra conv layers in G",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="z from bernoulli distribution, with prob=0.5",
    )
    return parser


def main(
    arg_list,
    project_name,
    run_name,
):
    parser = create_parser()
    opt = parser.parse_args(arg_list)
    print(opt)
    try:
        os.makedirs(opt.outDir)
    except OSError:
        pass

    set_seeds(opt)
    device = check_cuda()

    nc = 3
    ngpu = opt.ngpu
    nz = opt.nz
    ngf = opt.ngf
    ndf = opt.ndf
    n_extra_d = opt.n_extra_layers_d
    n_extra_g = opt.n_extra_layers_g

    # here opt becomes our core setting for this model to work .
    load_wandb(project_name, run_name, opt)

    image_folder = create_image_folder(
        create_dataset=False, use_default=True, image_size=opt.imageSize
    )

    data_loader = create_data_loader(
        image_folder,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.workers,
    )
    if active_logger:
        ic(data_loader)
        ic(nc, ngpu)
        ic(wandb.config)


if __name__ == "__main__":
    arg_list = [
        "--dataRoot",
        f"{hc.DIR}data/gallery-dl/anime-faces",
        "--workers",
        "16",
        "--batchSize",
        "128",
        "--imageSize",
        "64",
        "--nz",
        "100",
        "--ngf",
        "64",
        "--ndf",
        "64",
        "--niter",
        "80",
        "--lr",
        "0.0002",
        "--beta1",
        "0.5",
        "--cuda",
        "--ngpu",
        "1",
        "--netG",
        "",
        "--netD",
        "",
        "--outDir",
        "./results",
        "--model",
        "2",
        "--d_labelSmooth",
        "0.25",
        "--n_extra_layers_d",
        "0",
        "--n_extra_layers_g",
        "1",
    ]
    project_name = "SimpleAnimeGan"
    run_name = "test_run"
    main(arg_list, project_name, run_name)
