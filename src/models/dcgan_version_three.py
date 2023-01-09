import torch
import torch.nn as nn
import torch.nn.parallel

"""
This will have a z decoder and a fc layer.

Within the gan improvement paper, it said having a z decoder and a fc layer
will greatly improve performance.
"""


class DCGanVariantTwoGenerator(nn.Module):
    def __init__(self, gpu_number, nz, nc, ngf, n_extra_layers_g=None, leake_relu=0.2):
        self.gpu_number = gpu_number
        self.nz = nz
        # THE fc (fully connected layers)
        # https://arxiv.org/abs/1905.02417

        self.fcs = nn.Sequential(
            nn.Linear(nz, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.decode_fcs = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, nz),
        )

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(1024, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
