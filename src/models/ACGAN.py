import torch
from torch import nn

from src.core import hp


class Generator(nn.Module):
    """
    ACGANs with Illustration Vector generator.

    The ACGANs with Illustration Vector generator takes a noise vector, an illustration vector, and a class vector as input and generates an image.
    The generator is trained to generate not only realistic images, but also images that belong to a specific class.

    Attributes:
        latent_dim (int): the length of the noise vector
        illustration_dim (int): the length of the illustration vector
        class_dim (int): the length of the class vector (in one-hot form)
        gen (nn.Sequential): the main generator structure, a sequence of transpose convolutional layers, batch normalization layers, and activation functions

    Methods:
        __init__(self, latent_dim, class_dim): Initializes the ACGANs with Illustration Vector generator with the specified latent, illustration, and class dimensions.

    Todo :
        1. Generator requires : Extra layers, for conv, batchnorm and relu , to be another parser [Needs to be implemented]
        2. Discriminator requires : Extra layers as well, this will ne a remap it would have to be similar to how its done in dcgan version one . But with extended information
           `from .home.viv.GitHub.active_development.PROJECT.src.models import dcgan_version_one` Refere to this file for better import
    """

    def __init__(self, latent_dim, class_dim, extra_layers=1):
        """Initialize the Generator Class with latent_dim and class_dim.

        Args:
            latent_dim (int): the length of the noise vector
            class_dim (int): the length of the class vector (in one-hot form)
        """
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.class_dim = class_dim
        print("Class dim is ", self.class_dim)

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
        )

        if extra_layers > 0:
            for i in range(extra_layers):
                self.gen.add_module(
                    f"extra_conv_{i + 1}",
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                )
                self.gen.add_module(
                    f"extra_batchnorm_{i + 1}", nn.BatchNorm2d(128)
                )
                self.gen.add_module(
                    f"extra_relu_{i + 1}", nn.ReLU(inplace=True)
                )

        self.gen.add_module(
            "final_layer_deconv",
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
        )
        self.gen.add_module("final_layer_tanh", nn.Tanh())

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
        concat = torch.cat((noise, _class), dim=1)
        concat = concat.unsqueeze(2).unsqueeze(3)
        return self.gen(concat)


class Discriminator(nn.Module):
    """
    ACGANs with Illustration Vector discriminator.

    A discriminator that takes an image, an illustration vector, and a class vector as input and outputs a scalar indicating the probability that the image is real, and a vector of class probabilities indicating the class of the image.
    The discriminator is trained to not only distinguish the generated images from real images, but also to classify the generated images into the correct class.

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

    def __init__(self, hair_classes, eye_classes, extra_layers=0):
        """Initialize the Discriminator Class with the number of hair and eye classes.

        Args:
            hair_classes (int): the number of hair classes the discriminator needs to classify.
            eye_classes (int): the number of eye classes the discriminator needs to classify.
        """
        super(Discriminator, self).__init__()

        self.hair_classes = hair_classes
        self.eye_classes = eye_classes
        print("Discriminator")
        print("Hair classes: ", self.hair_classes)
        print("Eye classes: ", self.eye_classes)

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
        if extra_layers > 0:
            for i in range(extra_layers):
                self.bottleneck.add_module(
                    f"extra_conv_{i + 1}",
                    nn.Conv2d(1024, 1024, 3, 1, 1, bias=False),
                )
                self.bottleneck.add_module(
                    f"extra_batchnorm_{i + 1}", nn.BatchNorm2d(1024)
                )
                self.bottleneck.add_module(
                    f"extra_relu_{i + 1}", nn.LeakyReLU(0.2, inplace=True)
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
