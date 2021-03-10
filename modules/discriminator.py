import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


def resample_like(x, y, mode="bilinear"):
    x_resampled = F.interpolate(x, size=y.shape[-2:],
                                mode=mode,
                                align_corners=True if mode == "bilinear" else None)
    return x_resampled


class ConvNormAct(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, norm="bn",
                 padding_mode="reflect"):
        super(ConvNormAct, self).__init__()

        if norm is None:
            pass
        elif isinstance(norm, str):
            if norm.lower().strip() == "bn":
                norm_layer = nn.BatchNorm2d
            elif norm.lower().strip() == "in":
                norm_layer = nn.InstanceNorm2d
            elif norm.lower().strip() in ["none", ""]:
                pass
            else:
                raise ValueError(f"Unknown norm: {norm}")
        else:
            raise ValueError(f"Unknown norm: {norm}")

        self.block = nn.Sequential(
            *([nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation,
                         padding_mode=padding_mode), ]
              + ([norm_layer(out_channels), ]
                 if (norm and norm.lower().strip() not in ["none", ""]) else [])
              + [nn.LeakyReLU(0.2, inplace=True), ])
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, encoder_channels=[32, 64, 128, 256, 512]):
        super(UNet, self).__init__()

        basic_block = functools.partial(ConvNormAct,
                                        kernel_size=3, stride=1, padding=1,
                                        norm="bn")

        self.encoder_stage_1 = nn.Sequential(
            basic_block(in_channels, encoder_channels[0]),
            basic_block(encoder_channels[0], encoder_channels[0])
        )
        self.encoder_stage_2 = nn.Sequential(
            basic_block(encoder_channels[0], encoder_channels[1]),
            basic_block(encoder_channels[1], encoder_channels[1])
        )
        self.encoder_stage_3 = nn.Sequential(
            basic_block(encoder_channels[1], encoder_channels[2]),
            basic_block(encoder_channels[2], encoder_channels[2])
        )
        self.encoder_stage_4 = nn.Sequential(
            basic_block(encoder_channels[2], encoder_channels[3]),
            basic_block(encoder_channels[3], encoder_channels[3])
        )
        self.encoder_stage_5 = nn.Sequential(
            basic_block(encoder_channels[3], encoder_channels[4]),
            basic_block(encoder_channels[4], encoder_channels[4])
        )

        self.decoder_stage_1 = nn.Sequential(
            basic_block(
                encoder_channels[4] + encoder_channels[3], encoder_channels[3]),
            basic_block(encoder_channels[3], encoder_channels[3])
        )
        self.decoder_stage_2 = nn.Sequential(
            basic_block(
                encoder_channels[3] + encoder_channels[2], encoder_channels[2]),
            basic_block(encoder_channels[2], encoder_channels[2])
        )
        self.decoder_stage_3 = nn.Sequential(
            basic_block(
                encoder_channels[2] + encoder_channels[1], encoder_channels[1]),
            basic_block(encoder_channels[1], encoder_channels[1])
        )
        self.decoder_stage_4 = nn.Sequential(
            basic_block(
                encoder_channels[1] + encoder_channels[0], encoder_channels[0]),
            basic_block(encoder_channels[0], encoder_channels[0])
        )

        self.out_conv = nn.Conv2d(encoder_channels[0], out_channels, 3, 1, 1)

    def forward(self, x):
        x1 = self.encoder_stage_1(x)
        x2 = self.encoder_stage_2(F.max_pool2d(x1, (2, 2)))
        x3 = self.encoder_stage_3(F.max_pool2d(x2, (2, 2)))
        x4 = self.encoder_stage_4(F.max_pool2d(x3, (2, 2)))
        x5 = self.encoder_stage_5(F.max_pool2d(x4, (2, 2)))

        y1 = self.decoder_stage_1(
            torch.cat([resample_like(x5, x4), x4], dim=1))
        y2 = self.decoder_stage_2(
            torch.cat([resample_like(y1, x3), x3], dim=1))
        y3 = self.decoder_stage_3(
            torch.cat([resample_like(y2, x2), x2], dim=1))
        y4 = self.decoder_stage_4(
            torch.cat([resample_like(y3, x1), x1], dim=1))

        y = self.out_conv(y4)
        return y


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)