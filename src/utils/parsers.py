import argparse


def create_parser_varient_one():
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
        "--niter", type=int, default=2, help="number of epochs to train for"
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
