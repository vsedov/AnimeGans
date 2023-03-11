import torch
from torch import nn

# The model if you use the other dataset.


class Generator(nn.Module):
    """
    A class to define the Generator model of a Generative Adversarial Network (GAN).

    Attributes:
        latent_dim (int): The size of the latent space.
        class_dim (int): The size of the class space.
        relu_slope (float, optional): The slope of the negative part of the leaky ReLU activation function.
        generator (nn.Sequential): A sequential container to stack the transposed convolutional layers.
        class_embedding (nn.Sequential): A sequential container to stack the class embedding layers.
        feature_embedding (nn.Sequential): A sequential container to stack the feature embedding layers.
    """

    def __init__(self, latent_dim, class_dim, relu_slope=0.2):
        """
        The constructor of the `Generator` class.

        Args:
            latent_dim (int): The size of the latent space.
            class_dim (int): The size of the class space.
            relu_slope (float, optional): The slope of the negative part of the leaky ReLU activation function.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.class_dim = class_dim
        self.relu_scope = relu_slope

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024 + 256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=self.relu_slope, inplace=True),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=self.relu_slope, inplace=True),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=self.relu_slope, inplace=True),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )
        self.class_embedding = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.n_c,
                out_channels=256,
                kernel_size=4,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=self.relu_slope, inplace=True),
        )
        self.feature_embedding = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.latent_dim,
                out_channels=1024,
                kernel_size=4,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=self.relu_slope, inplace=True),
        )

    @property
    def device(self):
        """
        Returns the device that the model is on.

        Returns:
            torch.device: The device the model is on.
        """
        return next(self.parameters()).device

    def forward(self, inputs, class_vals):
        """
        The forward pass of the model.

        Args:
            inputs (torch.Tensor): The input tensor with shape `(batch_size, latent_dim)`.
            class_vals (torch.Tensor): The class values tensor with shape `(batch_size, class_dim)`.

        Returns:
            torch.Tensor: The output tensor with shape `(batch_size, 3, image_size, image_size)`.
        """
        feature_embedded = self.feature_embedding(
            inputs.unsqueeze(2).unsqueeze(3)
        )
        class_embedded = self.class_embedding(
            class_vals.unsqueeze(2).unsqueeze(3)
        )
        return self.generator(
            torch.cat((feature_embedded, class_embedded), dim=1)
        )


def modified_g_loss(fake_output, eps=1e-6):
    loss = (fake_output + eps).log().mean()
    return loss.neg()


# References:
# https://towardsdatascience.com/why-you-should-always-use-feature-embeddings-with-structured-datasets-7f280b40e716
