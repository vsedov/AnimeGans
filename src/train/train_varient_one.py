import argparse
import os
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from icecream import ic
from torch.autograd import Variable
from tqdm import tqdm

import wandb
from src.core import hc, hp
from src.data.anime import create_data_loader, create_image_folder
from src.models.dcgan_version_two import (
    DCGanVariantTwoDiscriminator,
    DCGanVariantTwoGenerator,
)
from src.utils.parsers import create_parser_varient_one
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


def main(
    arg_list,
    project_name,
    run_name,
):
    parser = create_parser_varient_one()
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
        ic(n_extra_d, n_extra_g)

    net_gen = DCGanVariantTwoGenerator(ngpu, nz, nc, ngf, leake_relu=0.2)
    net_disc = DCGanVariantTwoDiscriminator(ngpu, nz, nc, ndf)

    weights_init = hp.weights_init
    net_gen.apply(weights_init)
    if opt.netG != "":
        net_gen.load_state_dict(torch.load(opt.netG))
    print(net_gen)

    net_disc.apply(weights_init)
    if opt.netD != "":
        net_disc.load_state_dict(torch.load(opt.netD))
    print(net_disc)

    criterion = nn.BCELoss()
    criterion_MSE = nn.MSELoss()

    input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    if opt.binary:
        bernoulli_prob = torch.FloatTensor(opt.batchSize, nz, 1, 1).fill_(0.5)
        fixed_noise = torch.bernoulli(bernoulli_prob)
    else:
        fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    #  TODO: (vsedov) (15:52:08 - 21/02/23): Refactor this so i do not parse
    #  everything into cuda at once
    if opt.cuda:
        net_disc.cuda()
        net_gen.cuda()
        criterion.cuda()
        criterion_MSE.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    input = Variable(input)
    label = Variable(label)
    noise = Variable(noise)
    fixed_noise = Variable(fixed_noise)

    # setup optimizer
    optimizer_discrim = optim.Adam(
        net_disc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
    )
    optimizer_gen = optim.Adam(
        net_gen.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
    )
    #
    for epoch in range(opt.niter):
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            net_disc.zero_grad()
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            with torch.no_grad():
                input.resize_(real_cpu.size()).copy_(real_cpu)
            label.resize_(batch_size).fill_(real_label - opt.d_labelSmooth)
            output = net_disc(input)
            err_d_real = criterion(output, label.unsqueeze(1))
            err_d_real.backward()
            D_x = output.mean().item()
            noise.resize_(batch_size, nz, 1, 1)
            if opt.binary:
                bernoulli_prob.resize_(noise.size())
                noise.copy_(2 * (torch.bernoulli(bernoulli_prob) - 0.5))
            else:
                noise.normal_(0, 1)
            fake, z_prediction = net_gen(noise)
            label.fill_(fake_label)
            output = net_disc(fake.detach())
            err_d_fake = criterion(output, label.unsqueeze(1))
            err_d_fake.backward()  # gradients for fake/real will be accumulated
            D_G_z1 = output.mean().item()
            err_disc = err_d_real + err_d_fake
            optimizer_discrim.step()  # .step() can be called once the gradients are computed
            #  ╭────────────────────────────────────────────────────────────────────╮
            #  │                                                                    │
            #  │             (2) Update G network: maximize log(D(G(z)))            │
            #  │                                                                    │
            #  ╰────────────────────────────────────────────────────────────────────╯

            net_gen.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = net_disc(fake)
            err_gen = criterion(output, label.unsqueeze(1))
            err_gen.backward(
                retain_graph=True
            )  # True if backward through the graph for the second time
            # if opt.model == 2:  # with z predictor
            #
            #     err_gen_z = criterion_MSE(z_prediction, noise)
            #     err_gen_z.backward()

            D_G_z2 = output.mean().item()
            optimizer_gen.step()
            print(
                f"[{epoch}/{opt.niter}][{i}/{len(data_loader)}] Loss_D: {err_disc.data:.4f} Loss_G: {err_gen.data:.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f} "  # Elapsed: {elapsed_time:.2f}s
            )
            if i % 100 == 0:
                if not os.path.exists(opt.outDir):
                    os.makedirs(opt.outDir)
                vutils.save_image(
                    real_cpu[0:64, :, :, :],
                    "%s/real_samples.png" % opt.outDir,
                    nrow=8,
                )
                fake, _ = net_gen(fixed_noise)
                vutils.save_image(
                    fake.data[0:64, :, :, :],
                    "%s/fake_samples_epoch_varient_one%03d.png"
                    % (opt.outDir, epoch),
                    nrow=8,
                )
        if epoch % 1 == 0:
            torch.save(
                net_gen.state_dict(),
                "%s/netG_epoch_%d.pth" % (opt.outDir, epoch),
            )
            torch.save(
                net_disc.state_dict(),
                "%s/netD_epoch_%d.pth" % (opt.outDir, epoch),
            )


if __name__ == "__main__":
    arg_list = [
        "--dataRoot",
        f"{hc.DIR}data/gallery-dl/anime-faces",
        "--workers",
        "16",
        "--batchSize",
        "700",  # 128  Normal y
        "--imageSize",
        "64",
        "--nz",
        "100",
        "--ngf",
        "64",
        "--ndf",
        "64",
        "--niter",
        "2",  # 5 is for test purposes, as i want to see what i can get
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
        "2",  # here we would use the model two as the def ault model and template
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
