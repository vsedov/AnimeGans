class Generator(nn.Module):
    def __init__(self, latent_dim, class_dim, extra_layers=0):
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
        if self.extra_layers > 0:
            for i in range(self.extra_layers):
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
