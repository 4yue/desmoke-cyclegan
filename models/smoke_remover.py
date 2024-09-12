"""
Encoder-decoder networks for Smoke Remover
"""
import torch.nn as nn
import torch.nn.functional as F

# todo

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class SmokeRemover(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(SmokeRemover, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.enc = nn.Sequential(*model)

        # Residual blocks
        model = [ResidualBlock(in_features)]
        for _ in range(n_residual_blocks - 1):
            model += [ResidualBlock(in_features)]
        self.mid = nn.Sequential(*model)

        # Upsampling
        out_features = in_features//2
        model = [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)]
        in_features = out_features
        out_features = in_features // 2
        model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(out_features),
                  nn.ReLU(inplace=True)]

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.dec =nn.Sequential(*model)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)