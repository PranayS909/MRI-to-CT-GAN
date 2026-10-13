import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True, use_dropout=False):
        super().__init__()
        if down:
            layers = [
                nn.Conv2d(in_c, out_c, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        else:
            layers = [
                nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self, in_c=1, out_c=1, ngf=64):
        super().__init__()
        self.down0 = nn.Sequential(
            nn.Conv2d(in_c, ngf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down1 = UNetBlock(ngf, ngf * 2, down=True)
        self.down2 = UNetBlock(ngf * 2, ngf * 4, down=True)
        self.down3 = UNetBlock(ngf * 4, ngf * 8, down=True)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up0 = UNetBlock(ngf * 8, ngf * 8, down=False, use_dropout=True)
        self.up1 = UNetBlock(ngf * 16, ngf * 4, down=False)
        self.up2 = UNetBlock(ngf * 8, ngf * 2, down=False)
        self.up3 = UNetBlock(ngf * 4, ngf, down=False)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_c, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d0 = self.down0(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        bn = self.bottleneck(d3)
        u0 = self.up0(bn)
        u1 = self.up1(torch.cat([u0, d3], dim=1))
        u2 = self.up2(torch.cat([u1, d2], dim=1))
        u3 = self.up3(torch.cat([u2, d1], dim=1))
        return self.final(torch.cat([u3, d0], dim=1))

class Discriminator(nn.Module):
    def __init__(self, in_c=2, ndf=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)
        )

    def forward(self, mri, ct):
        return self.model(torch.cat([mri, ct], dim=1))