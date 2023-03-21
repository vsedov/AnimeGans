import torch
from torch import nn

from src.core import hp


class DCGanVariantOneGenerator(nn.Module):
    """Generator for the DCGAN variant one architecture.

    Args:
        gpu_count (int): Number of GPUs to use for parallel processing.
        noise_vector_size (int): Size of the noise vector to generate an image from.
        output_channels (int): Number of channels in the generated image.
        feature_maps (int): Number of feature maps in the generator network.
        extra_layers (int, optional): Number of extra convolutional layers to add to the generator network.
        leaky_relu_slope (float, optional): Slope for the leaky ReLU activation function.

    Attributes:
        gen (nn.Sequential): The generator network.
    """

    def __init__(
        self,
        gpu_count,
        noise_vector_size,
        output_channels,
        feature_maps,
        extra_layers=None,
        leaky_relu_slope=0.2,
    ):
        super().__init__()
        self.gpu_count = gpu_count
        self.gen = nn.Sequential(
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.ConvTranspose2d(
                feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.ConvTranspose2d(
                feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.ConvTranspose2d(
                feature_maps * 2, feature_maps, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
        )

        if extra_layers is not None:
            for i in range(extra_layers):
                self.gen.add_module(
                    f"extra_layer_{i}_conv",
                    nn.Conv2d(feature_maps, feature_maps, 3, 1, 1, bias=False),
                )
                self.gen.add_module(
                    f"extra_layer_{i}_batchnorm", nn.BatchNorm2d(feature_maps)
                )
                self.gen.add_module(
                    f"extra_layer_{i}_relu",
                    nn.LeakyReLU(leaky_relu_slope, inplace=True),
                )

        self.gen.add_module(
            "final_layer_deconv",
            nn.ConvTranspose2d(
                feature_maps, output_channels, 4, 2, 1, bias=False
            ),
        )
        self.gen.add_module("final_layer_tanh", nn.Tanh())

    def forward(self, input):
        """Forward pass of the generator.

        Args:
            input (torch.Tensor): The noise vector to generate an image from.

        Returns:
            torch.Tensor: The generated image.
        """
        gpu_ids = None
        if (
            isinstance(input.data, torch.cuda.FloatTensor)
            and self.gpu_count > 1
        ):
            gpu_ids = range(self.gpu_count)
        return nn.parallel.data_parallel(self.gen, input, gpu_ids)


class DCGanVariantOneDiscriminator(nn.Module):
    """Discriminator for the DCGAN variant one architecture.

    Args:
        gpu_count (int): Number of GPUs to use for parallel processing.
        input_channels (int): Number of channels in the input image.
        feature_maps (int): Number of feature maps in the discriminator network.
        extra_layers (int): Number of extra convolutional layers to add to the discriminator network.
        leaky_relu_slope (float, optional): Slope for the leaky ReLU activation function.

    Attributes:
        disc (nn.Sequential): The discriminator network.
    """

    def __init__(
        self,
        gpu_count,
        input_channels,
        feature_maps,
        extra_layers,
        leaky_relu_slope=0.2,
    ):
        super().__init__()
        self.gpu_count = gpu_count
        self.disc = nn.Sequential(
            nn.Conv2d(input_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
        )

        for i in range(extra_layers):
            self.disc.add_module(
                f"extra_layers_{i}_{feature_maps * 4}_conv",
                nn.Conv2d(
                    feature_maps * 4, feature_maps * 4, 3, 1, 1, bias=False
                ),
            )
            self.disc.add_module(
                f"extra_layers_{i}_{feature_maps * 4}_batchnorm",
                nn.BatchNorm2d(feature_maps * 4),
            )
            self.disc.add_module(
                f"extra_layers_{i}_{feature_maps * 4}_relu",
                nn.LeakyReLU(leaky_relu_slope, inplace=True),
            )

        self.disc.add_module(
            "final_layer_conv",
            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False),
        )
        self.disc.add_module("final_layer_sigmoid", nn.Sigmoid())

    def forward(self, input):
        """Forward pass of the discriminator.

        Args:
            input (torch.Tensor): The input image to classify as real or fake.

        Returns:
            torch.Tensor: The discriminator output.
        """
        gpu_ids = None
        if input.is_cuda and self.gpu_count > 1:
            gpu_ids = range(self.gpu_count)
        output = nn.parallel.data_parallel(self.disc, input, gpu_ids)
        return output.view(-1, 1)


def evaluation():
    print(f"nz = {hp.get_core('nz')}")
    print(f"nc = {hp.get_core('nc')}")
    print(f"ngf = {hp.get_core('ngf')}")
    print(f"extra_layers_g = {hp.get_core('extra_layers_g')}")
    print(f"extra_layers_d = {hp.get_core('extra_layers_d')}")
    print(f"ndf = {hp.get_core('ndf')}")

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
        DCGanVariantOneDiscriminator(
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
