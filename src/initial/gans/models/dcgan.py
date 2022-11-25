from icecream import ic
from torch import nn

from src.core import hc, hp


class Generator(nn.Module):
    def __init__(self, n_z, n_f, n_c, relu_slope=0.2):
        """Generator Init
        n_z : Latent Vector Size
            In channel data points for latent Vector
        n_f : Image Size
            Size of the images that we have.
        n_c : Channel Size
            In this situation the nc is the output channels
        relu_slope : Relu : we use leaky relu as it provides more fruitful results
            this can be any size given a given range of 0 and 1 , most optimal tends to be around 0.2.
        """
        super().__init__()

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(in_channels=n_z, out_channels=n_f * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_f * 8),
            nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
            nn.ConvTranspose2d(
                in_channels=n_f * 8, out_channels=n_f * 4, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(n_f * 4),
            nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
            nn.ConvTranspose2d(
                in_channels=n_f * 4, out_channels=n_f * 2, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(n_f * 2),
            nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
            nn.ConvTranspose2d(in_channels=n_f * 2, out_channels=n_f, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_f),
            nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
            nn.ConvTranspose2d(in_channels=n_f, out_channels=n_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, inputs):
        return self.generator(inputs)


class Discriminator(nn.Module):
    def __init__(self, n_f, n_c, relu_slope=0.2):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=n_c, out_channels=n_f, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
            nn.Conv2d(in_channels=n_f, out_channels=n_f * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_f * 2),
            nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
            nn.Conv2d(in_channels=n_f * 2, out_channels=n_f * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_f * 4),
            nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
            nn.Conv2d(in_channels=n_f * 4, out_channels=n_f * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_f * 8),
            nn.LeakyReLU(negative_slope=relu_slope, inplace=True),
            nn.Conv2d(in_channels=n_f * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, inputs):
        return self.discriminator(inputs)


def show_info():

    params = hc.initial["params"]

    gen = Generator(params["latent_vector"], params["generator"], params["output"])
    dis = Discriminator(params["discriminator"], params["output"])

    hp.sum(gen)
    hp.sum(dis)
    print("\n")
    ic(gen)
    ic(dis)


if __name__ == "__main__":
    exit(show_info())
else:
    pass
