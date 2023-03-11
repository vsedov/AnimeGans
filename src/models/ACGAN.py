import torch
from torch import nn

from src.core import hp


class Generator(nn.Module):
    """Adversarial Conditional Generative Adversarial Network (ACGAN) generator.

    The ACGAN generator takes a noise vector and a class vector as input and generates an image.
    The generator is based on the architecture described in the 2016 DCGAN paper, with activation functions and batch
        normalization following the same structure.

    Attributes:
        latent_dim (int): the length of the noise vector
        class_dim (int): the length of the class vector (in one-hot form)
        gen (nn.Sequential): the main generator structure, a sequence of convolutional transpose layers, batch
            normalization layers, and activation functions
    """

    def __init__(self, latent_dim, class_dim):
        """Initialize the Generator Class with latent_dim and class_dim.

        Args:
            latent_dim (int): the length of the noise vector
            class_dim (int): the length of the class vector (in one-hot form)
        """
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.class_dim = class_dim

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.latent_dim + self.class_dim,
                out_channels=1024,
                kernel_size=4,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    @property
    def device(self):
        """Return the device of the generator."""
        return next(self.parameters()).device

    def forward(self, noise, _class):
        """Define the forward pass of the Generator Class.

        Args:
            noise (torch.Tensor): the input noise vector
            _class (torch.import torchensor): the input class vector. The vector need not be one-hot since multilabel
                generation is supported.
        Returns:
            The generated image (torch.Tensor).
        """
        concat = torch.cat(
            (noise, _class), dim=1
        )  # Concatenate noise and class vector.
        concat = concat.unsqueeze(2).unsqueeze(
            3
        )  # Reshape the latent vector into a feature map.
        return self.gen(concat)


class Discriminator(nn.Module):
    """Adversarial Conditional Generative Adversarial Network (ACGAN) discriminator.

    A modified version of the Deep Convolutional Generative Adversarial Network (DCGAN) discriminator.
    In addition to the discriminator output, the ACGAN discriminator also classifies the class of the input image
    using a fully-connected layer.

    Attributes:
        hair_classes (int): the number of hair classes the discriminator needs to classify.
        eye_classes (int): the number of eye classes the discriminator needs to classify.
        conv_layers (nn.Sequential): all convolutional layers before the last DCGAN layer, used as a feature extractor.
        discriminator_layer (nn.Sequential): the last layer of the DCGAN that outputs a single scalar.
        bottleneck (nn.Sequential): the layer before the classifier layers.
        hair_classifier (nn.Sequential): the fully connected layer for hair class classification.
        eye_classifier (nn.Sequential): the fully connected layer for eye class classification.
    """

    def __init__(self, hair_classes, eye_classes):
        """Initialize the Discriminator Class with the number of hair and eye classes.

        Args:
            hair_classes (int): the number of hair classes the discriminator needs to classify.
            eye_classes (int): the number of eye classes the discriminator needs to classify.
        """
        super(Discriminator, self).__init__()

        self.hair_classes = hair_classes
        self.eye_classes = eye_classes

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.discriminator_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=1024, out_channels=1, kernel_size=4, stride=1
            ),
            nn.Sigmoid(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=1024, out_channels=1024, kernel_size=4, stride=1
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )
        self.hair_classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, self.hair_classes),
            nn.Softmax(),
        )

        self.eye_classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, self.eye_classes),
            nn.Softmax(),
        )

        return

    @property
    def device(self):
        """Return the device of the discriminator."""
        return next(self.parameters()).device

    def forward(self, _input):
        """Define the forward pass of the discriminator.

        Args:
            image (torch.Tensor): a batch of image tensors with shape (N, 3, 64, 64).

        Returns:
            discrim_output (torch.Tensor): a tensor with values between 0-1 indicating if the image is real or fake-Shape: (N, 1)
            hair_class (torch.Tensor): class scores for each hair class. Shape: (N, hair_classes).
            eye_class (torch.Tensor): class scores for each eye class. Shape: (N, eye_classes).
        """
        features = self.conv_layers(_input)
        discrim_output = self.discriminator_layer(features).view(
            -1
        )  # Single-value scalar

        flatten = self.bottleneck(features).squeeze()
        hair_class = self.hair_classifier(
            flatten
        )  # Outputs probability for each hair class label
        eye_class = self.eye_classifier(
            flatten
        )  # Outputs probability for each eye class label
        return discrim_output, hair_class, eye_class


if __name__ == "__main__":
    latent_dim = 128
    class_dim = 22
    batch_size = 2
    z = torch.randn(batch_size, latent_dim)
    c = torch.randn(batch_size, class_dim)

    G = Generator(latent_dim, class_dim)
    D = Discriminator(12, 10)

    hp.show_sum(G)
    hp.show_sum(D)

    print((G(z, c)).shape)
    x, y, z = D(G(z, c))
    print(x.shape, y.shape, z.shape)
