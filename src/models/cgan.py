import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, n_z, n_c, relu_slope=0.2):
        super().__init__()
        self.n_z = n_z
        self.classifier = n_c
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
                in_channels=self.n_z,
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
        return next(self.parameters()).device

    def forward(self, inputs, class_vals):
        return self.generator(
            torch.cat(
                (
                    self.feature_embedding(
                        inputs.unsqueeze(2).unsqueeze(3),
                        class_vals.unsqueeze(2).unsqueeze(3),
                    )
                ),
                dim=1,
            )
        )


def modified_g_loss(fake_output, eps=1e-6):
    loss = (fake_output + eps).log().mean()
    return loss.neg()


# References:
# https://towardsdatascience.com/why-you-should-always-use-feature-embeddings-with-structured-datasets-7f280b40e716
