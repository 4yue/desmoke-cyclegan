import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_nc, hidden_nc, n_layers=3):
        super(Encoder, self).__init__()

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(inplace=True)]

        in_features = 64
        out_features = in_features * 2
        for _ in range(n_layers - 1):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2

        model += [nn.Conv2d(in_features, hidden_nc, 3, stride=2, padding=1), nn.InstanceNorm2d(64), nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, hidden_nc, output_nc, n_layers=3):
        super(Decoder, self).__init__()

        in_features = hidden_nc
        out_features = output_nc // 2

        model = []
        for _ in range(n_layers):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Mid_Xnet(nn.Module):
    def __init__(self, features, n_layers):
        super(Mid_Xnet, self).__init__()

        model = [ResidualBlock(features)]
        for _ in range(n_layers - 1):
            model += [ResidualBlock(features)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class SmokeAdder(nn.Module):
    def __init__(self, input_nc, output_nc, hidden_nc=512, n_residual_block=9):
        super(SmokeAdder, self).__init__()

        self.enc = Encoder(input_nc, hidden_nc)
        self.mid = Mid_Xnet(hidden_nc, n_residual_block)
        self.dec = Decoder(hidden_nc, output_nc)

    def forward(self, x):
        x = self.enc(x)
        x = self.mid(x)
        x = self.dec(x)
        return x


class SmokeRemover(nn.Module):
    def __init__(self, input_nc, output_nc, hidden_nc=512, n_residual_block=9):
        super(SmokeRemover, self).__init__()

        self.enc1 = Encoder(input_nc, hidden_nc // 2)
        self.enc2 = Encoder(input_nc, hidden_nc // 2)

        self.mid = Mid_Xnet(hidden_nc, n_residual_block)
        # self.mid = Mid_Xnet(hidden_nc * 2, n_residual_block)

        self.dec = Decoder(hidden_nc, output_nc)

    def forward(self, x1, x2):
        x1 = self.enc1(x1)
        x2 = self.enc2(x2)
        x = self.mid(torch.cat([x1, x2], dim=1))
        x = self.dec(x)
        return x


