import torch

from src.core import hp


class DCGanVariantOneGenerator(nn.Module):
    def __init__(
        self, gpu_number, nz, nc, ngf, n_extra_layers_g=None, leaky_relu=0.2
    ):
        super().__init__()
        self.gpu_number = gpu_number
        self.gen = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(leaky_relu, inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(leaky_relu, inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(leaky_relu, inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(leaky_relu, inplace=True),
        )

        if n_extra_layers_g is not None:
            for t in range(n_extra_layers_g):
                self.gen.add_module(
                    f"extra-layers-{t}-{ngf}-conv",
                    nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
                )
                self.gen.add_module(
                    f"extra-layers_{t}_{ngf}_batchnorm", nn.BatchNorm2d(ngf)
                )
                self.gen.add_module(
                    f"extra-layers_{t}_{ngf}_relu",
                    nn.LeakyReLU(0.2, inplace=True),
                )

        self.gen.add_module(
            "final_layer_deconv",
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        )  # 5,3,1 for 96x96
        self.gen.add_module("final_layer_tanh", nn.Tanh())

    def forward(self, input):
        gpu_ids = None
        # I wonder if you use the default device, would this of had a bigger
        # impact or not (∩⌣̀_⌣́)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.gpu_number)
        return nn.parallel.data_parallel(self.gen, input, gpu_ids)


class DCGanVariantOneDiscriminator(nn.Module):
    def __init__(
        self, gpu_number, nz, nc, ndf, n_extra_layers_d, leaky_relu=0.2
    ):
        super().__init__()
        self.gpu_number = gpu_number
        self.disc = nn.Sequential(
            # 5,3,1 for 96x96
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(leaky_relu, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(leaky_relu, inplace=True),
            # state size. (ndf*2) x 16 x 16
            # nn.Coself, gpu_number, nz, nc, ndf, leaky_relu=0.2):      nn.LeakyReLU(leaky_relu, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(leaky_relu, inplace=True),
        )
        for i in range(n_extra_layers_d):
            self.disc.add_module(
                f"fextra_layers_{i}_{ndf*8}_conv",
                nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
            )
            self.disc.add_module(
                f"fextra_layers_{i}_{ndf*8}_batchnorm", nn.BatchNorm2d(ndf * 8)
            )
            self.disc.add_module(
                f"fextra_layers_{i}_{ndf*8}_relu",
                nn.LeakyReLU(leaky_relu, inplace=True),
            )

        self.disc.add_module(
            "final_layers_conv", nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )
        self.disc.add_module("final_layers_sigmoid", nn.Sigmoid())

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            nn.parallel.data_parallel(self.convs, input, gpu_ids).view(-1, 1024)

        return output.view(-1, 1)



def evaluation():
    print("Generator With Extra Layers\n")
    hp.show_sum(
        DCGanVariantOneGenerator(
            1,
            hp.get_core("nz"),
            hp.get_core("nc"),
            hp.get_core("ngf"),
            hp.get_core("extra_layers_g"),
        )
    )

    hp.show_sum(
        DCGanVariantTwoDiscriminator(
            1,
            hp.get_core("nz"),
            hp.get_core("nc"),
            hp.get_core("ngf"),
        )
    )        DCGanVariantOneDiscriminator(
            1,
            hp.get_core("nz"),
            hp.get_core("nc"),
            hp.get_core("ndf"),
            hp.get_core("extra_layers_d"),
        )
    )


if __name__ == "__main__":
    exit(evaluation())
else:
    pass
