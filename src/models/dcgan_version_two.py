import torch
from torch import nn

from src.core import hp

"""
This will have a z decoder and a fc layer.

Within the gan improvement paper, it said having a z decoder and a fc layer
will greatly improve performance.
"""


class DCGanVariantTwoGenerator(nn.Module):
    """
    A class to define the generator model of a Deep Convolutional Generative Adversarial Network (DC-GAN) variant.

    Attributes:
        gpu_number (int): The number of GPUs available to the model.
        latent_dim (int): The size of the latent space.
        num_channels (int): The number of channels in the generated image.
        num_filters (int): The number of filters in the convolutional layers.
        leake_relu (float, optional): The slope of the negative part of the leaky ReLU activation function.
        fcs (nn.Sequential): A sequential container to stack the fully connected layers.
        decode_fcs (nn.Sequential): A sequential container to stack the fully connected layers for decoding.
        convs (nn.Sequential): A sequential container to stack the transposed convolutional layers.
    """

    def __init__(
        self, gpu_number, latent_dim, num_channels, num_filters, leake_relu=0.2
    ):
        """
        The constructor of the `DCGanVariantTwoGenerator` class.

        Args:
            gpu_number (int): The number of GPUs available to the model.
            latent_dim (int): The size of the latent space.
            num_channels (int): The number of channels in the generated image.
            num_filters (int): The number of filters in the convolutional layers.
            leake_relu (float, optional): The slope of the negative part of the leaky ReLU activation function.
        """
        super().__init__()
        self.gpu_number = gpu_number
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.leake_relu = leake_relu

        self.fcs = nn.Sequential(
            nn.Linear(latent_dim, 1024),
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
            nn.Linear(1024, latent_dim),
        )

        self.convs = nn.Sequential(
            nn.ConvTranspose2d(1024, num_filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                num_filters * 8, num_filters * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                num_filters * 4, num_filters * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                num_filters * 2, num_filters, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters, num_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        """
        The forward pass of the model.

        Args:
            input (torch.Tensor): The input tensor with shape `(batch_size, latent_dim)`.

        Returns:
            tuple: A tuple containing the output tensor with shape `(batch_size, num_channels, image_size, image_size)` and
            the tensor representing the decoded latent space with shape `(batch_size, latent_dim)`.
        """
        input = self.fcs(input.view(-1, self.latent_dim))
        gpu_ids = None
        if (
            isinstance(input.data, torch.cuda.FloatTensor)
            and self.gpu_number > 1
        ):
            gpu_ids = range(self.gpu_number)
        decoded_latent = self.decode_fcs(input)
        input = input.view(-1, 1024, 1, 1)
        generated_image = nn.parallel.data_parallel(self.convs, input, gpu_ids)
        return generated_image, decoded_latent


class DCGanVariantTwoDiscriminator(nn.Module):
    def __init__(self, gpu_number, num_channels, num_filters, leaky_relu=0.2):
        """
        Initializes the DCGanVariantTwoDiscriminator.

        Args:
            gpu_number (int): The number of GPUs to use for the model.
            num_channels (int): The number of channels in the input image.
            num_filters (int): The number of filters in the first convolutional layer.
            leaky_relu (float): The negative slope for the leaky ReLU activation.
        """
        super().__init__()
        self.gpu_number = gpu_number
        self.convs = nn.Sequential(
            nn.Conv2d(num_channels, num_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(leaky_relu, inplace=True),
            nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(leaky_relu, inplace=True),
            nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(leaky_relu, inplace=True),
            nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(leaky_relu, inplace=True),
            nn.Conv2d(num_filters * 8, 1024, 4, 1, 0, bias=False),
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
        """
        The forward pass of the model.

        Args:
            input (torch.Tensor): The input tensor with shape `(batch_size, num_channels, image_size, image_size)`.

        Returns:
            torch.Tensor: The output tensor with shape `(batch_size, 1)`.
        """
        gpu_ids = None
        if (
            isinstance(input.data, torch.cuda.FloatTensor)
            and self.gpu_number > 1
        ):
            gpu_ids = range(self.gpu_number)
        output = nn.parallel.data_parallel(self.convs, input, gpu_ids)
        output = self.fcs(output.view(-1, 1024))
        return output.view(-1, 1)


def evaluation():
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
