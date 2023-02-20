import torch
import torch.nn as nn
import torch.nn.parallel

from src.core import hp

"""
This will have a z decoder and a fc layer.

Within the gan improvement paper, it said having a z decoder and a fc layer
will greatly improve performance.
"""


class DCGanVariantTwoGenerator(nn.Module):
    def __init__(self, gpu_number, nz, nc, ngf, leake_relu=0.2):
        super().__init__()
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

    def forward(self, input):
        input = self.fcs(input.view(-1, self.nz))
        gpu_ids = None
        # I wonder if you use the default device, would this of had a bigger
        # impact or not (∩⌣̀_⌣́)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.gpu_number)

        prediction_z = self.decode_fcs(input)
        return (
            nn.parallel.data_parallel(
                self.conv, input.view(-1, 1024, 1, 1), gpu_ids
            ),
            prediction_z,
        )


class DCGanVariantTwoDiscriminator(nn.Module):
    def __init__(self, gpu_number, nz, nc, ndf, leaky_relu=0.2):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1024, 4, 1, 0, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.fcs = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.gpu_number)
        output = self.fcs(
            nn.parallel.data_parallel(self.convs, input, gpu_ids).view(-1, 1024)
        )
        return output.view(-1, 1)


def evaluation():
    print(
        "Generator with Fully connected Layers; which is well known for training\n\n"
    )

    hp.show_sum(
        DCGanVariantTwoGenerator(
            1,
            hp.get_core("nz"),
            hp.get_core("nc"),
            hp.get_core("ngf"),
        )
    )

    hp.show_sum(
        DCGanVariantTwoDiscriminator(
            1,
            hp.get_core("nz"),
            hp.get_core("nc"),
            hp.get_core("ngf"),
        )
    )


if __name__ == "__main__":
    exit(evaluation())
else:
    pass
