import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, padding=1):
        super().__init__()
        model = [nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=padding, padding_mode='reflect'),
                 nn.InstanceNorm2d(out_channels),
                 nn.ReLU(inplace=True)]
        model += [nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=padding, padding_mode='reflect'),
                 nn.InstanceNorm2d(out_channels),
                 nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + self.model(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, res_blocks=6):
        super().__init__()
        self.res_blocks = res_blocks
        model = [
            nn.Conv2d(in_channels, 64, 7, padding_mode='reflect', padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True) 
        ]
        model += [
            nn.Conv2d(64, 128, 3, 2, padding_mode='reflect', padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        ]
        model += [
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        ]

        model += [ResidualBlock(256, 256) for _ in range(res_blocks)]

        model += [
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        ]
        model += [
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        model += [
            nn.Conv2d(64, out_channels, 7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        model = [
            nn.Conv2d(in_channels, 64, 4, 2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        model += [
            nn.Conv2d(64, 128, 4, 2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, True)
        ]
        model += [
            nn.Conv2d(128, 256, 4, 2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, True)
        ]
        model += [
            nn.Conv2d(256, 512, 4, 2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, True)
        ]
        model += [nn.Conv2d(512, 1, 4, stride=1, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x.mean().view([1])