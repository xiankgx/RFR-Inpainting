import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision import models

from modules.partialconv2d_nvidia import PartialConv2d

epsilon = 1e-6

###############################################################################
# Basic blocks
###############################################################################


class ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1,
                 padding_mode="reflect",
                 ReLU=nn.LeakyReLU):
        super(ConvBNReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation=dilation,
                      bias=False,
                      padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1,
                 padding_mode="reflect",
                 ReLU=nn.LeakyReLU):
        super(UpConvBNReLU, self).__init__()
        self.block = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),

            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      dilation=dilation,
                      bias=False,
                      padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding,
                      dilation=dilation,
                      bias=False,
                      padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class PConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 multi_channel=True,
                 relu_fn=F.leaky_relu):
        super(PConvBNReLU, self).__init__()
        self.pconv = PartialConv2d(in_channels, out_channels, kernel_size, stride, padding,
                                   bias=False,
                                   multi_channel=multi_channel,  # multi-channel mask
                                   return_mask=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = relu_fn

    def forward(self, image, mask):
        x, mask = self.pconv(image, mask)
        return self.relu(self.bn(x), inplace=True), mask


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes,
                               kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes,
                               kernel_size=3, stride=stride, padding=1,
                               bias=False,
                               padding_mode="reflect")
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """

        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer,)


class KnowledgeConsistentAttention(nn.Module):
    def __init__(self, patch_size=1, propagate_size=3, stride=1):
        super(KnowledgeConsistentAttention, self).__init__()

        self.patch_size = patch_size
        self.propagate_size = propagate_size
        # self.stride = stride  # unused
        self.prop_kernels = None

        self.att_scores_prev = None
        self.masks_prev = None
        self.ratio = nn.Parameter(torch.ones(1))

    def reset(self):
        self.att_scores_prev = None
        self.masks_prev = None

    def forward(self, foreground, masks):
        bz, nc, h, w = foreground.size()
        if masks.size(3) != foreground.size(3):
            masks = F.interpolate(masks, foreground.size()[2:])

        background = foreground.clone()
        conv_kernels_all = background.view(bz, nc, w * h, 1, 1)
        conv_kernels_all = conv_kernels_all.permute(0, 2, 1, 3, 4)
        # print(f"conv_kernels_all.shape: {conv_kernels_all.shape}")
        output_tensor = []
        att_score = []

        for i in range(bz):
            feature_map = foreground[i:i+1]
            # print(f"feature_map.shape: {feature_map.shape}")

            conv_kernels = conv_kernels_all[i] + epsilon
            norm_factor = torch.sum(conv_kernels ** 2, [1, 2, 3],
                                    keepdim=True) ** 0.5
            conv_kernels = conv_kernels/norm_factor
            # print(f"conv_kernels.shape: {conv_kernels.shape}")

            conv_result = F.conv2d(feature_map, conv_kernels,
                                   padding=self.patch_size//2
                                   )
            # print(f"conv_result.shape: {conv_result.shape}")
            if self.propagate_size != 1:
                conv_result = F.avg_pool2d(conv_result, 3, 1, padding=1) * 9

            attention_scores = F.softmax(conv_result, dim=1)
            # print(f"attention_scores.shape: {attention_scores.shape}")

            if self.att_scores_prev is not None:
                attention_scores = \
                    (self.att_scores_prev[i:i+1] * self.masks_prev[i:i+1] + attention_scores * (torch.abs(self.ratio) + epsilon)) \
                    / (self.masks_prev[i:i+1] + (torch.abs(self.ratio) + epsilon))
            att_score.append(attention_scores)

            feature_map = F.conv_transpose2d(attention_scores, conv_kernels,
                                             stride=1,
                                             padding=self.patch_size//2
                                             )
            final_output = feature_map
            output_tensor.append(final_output)

        self.att_scores_prev = torch.cat(att_score, dim=0).view(bz, h*w, h, w)
        self.masks_prev = masks.view(bz, 1, h, w)
        return torch.cat(output_tensor, dim=0)


class _KnowledgeConsistentNonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        """
        :param in_channels:
        :param inter_channels:
        :param dimension:
        :param sub_sample:
        :param bn_layer:
        """

        super(_KnowledgeConsistentNonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

        # Knowledge consistent
        self.lamb = nn.Parameter(torch.zeros(1))
        self.initialize_knowledge()

        # Attention map smoothing
        self.smooth_attention_map = True
        self.smooth_atention_map_kernel_size = 3
        self.lamb_smooth = nn.Parameter(torch.zeros(1))

    def initialize_knowledge(self):
        self.nl_map_prev = None
        self.mask_prev = None

    def forward(self, x, mask, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask, x.shape[-2:],
                                 mode="nearest")

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        assert not torch.isinf(theta_x).any()
        assert not torch.isinf(phi_x).any()
        assert not torch.isnan(theta_x).any()
        assert not torch.isnan(phi_x).any()

        f = torch.matmul(theta_x, phi_x)
        # input = torch.zeros(batch_size, x.shape[-2] * x.shape[-1], self.inter_channels).type(torch.float32).to(x)
        # f = torch.baddbmm(input, theta_x, phi_x,
        #                  beta=0, alpha=1, out=None)
        # print(f.shape)

        assert not torch.isinf(f).any(), f.dtype

        if self.smooth_attention_map:
            f_smooth = F.avg_pool2d(f, self.smooth_atention_map_kernel_size,
                                    stride=1,
                                    padding=(
                                        self.smooth_atention_map_kernel_size - 1) // 2,
                                    count_include_pad=False)
            lamb_smooth = torch.sigmoid(self.lamb_smooth)
            f = lamb_smooth * f + (1.0 - lamb_smooth) * f_smooth

        f_div_C = F.softmax(f, dim=-1)
        if self.nl_map_prev is not None:
            lamb = torch.sigmoid(self.lamb)

            # print(f"lamb: {lamb}")
            # print(f"f_div_C: {f_div_C.shape}")
            # print(f"self.nl_map_prev: {self.nl_map_prev.shape}")
            # print(f"self.mask_prev: {self.mask_prev.shape}")
            # print(f"g_x: {g_x.shape}")

            mask_prev = self.mask_prev \
                .view(*self.mask_prev.shape[:2], -1) \
                .permute(0, 2, 1)

            f_div_C = (lamb * f_div_C + (1.0 - lamb) * (self.nl_map_prev * mask_prev)) \
                / (lamb + (1.0 - lamb) * mask_prev)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        # Save state
        self.nl_map_prev = f_div_C
        self.mask_prev = mask

        if return_nl_map:
            return z, f_div_C
        return z


class KnowledgeConsistentNONLocalBlock2D(_KnowledgeConsistentNonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(KnowledgeConsistentNONLocalBlock2D, self).__init__(in_channels,
                                                                 inter_channels=inter_channels,
                                                                 dimension=2, sub_sample=sub_sample,
                                                                 bn_layer=bn_layer,)


class AttentionModule(nn.Module):

    def __init__(self, inchannel, att_module=NONLocalBlock2D):
        super(AttentionModule, self).__init__()
        self.att = att_module(inchannel)

        self.knowledge_consistent_attention = isinstance(
            self.att, KnowledgeConsistentNONLocalBlock2D)

        self.combiner = nn.Conv2d(inchannel * 2, inchannel, kernel_size=1)

    def forward(self, foreground, mask):
        if self.knowledge_consistent_attention:
            outputs = self.att(foreground, mask)
        else:
            outputs = self.att(foreground)
        outputs = torch.cat([outputs, foreground], dim=1)
        outputs = self.combiner(outputs)
        return outputs

    def initialize_knowledge(self):
        if self.knowledge_consistent_attention:
            self.att.initialize_knowledge()


class RFRModule(nn.Module):

    def __init__(self, in_channel=64, layer_size=6, att_module=NONLocalBlock2D):
        super(RFRModule, self).__init__()
        self.layer_size = layer_size

        # Down convs 3x
        for i in range(3):
            name = 'enc_{:d}'.format(i + 1)
            out_channel = in_channel * 2
            block = ConvBNReLU(in_channel, out_channel, 3, 2, 1)
            in_channel = out_channel
            setattr(self, name, block)

        # Dilated convs
        for i in range(3, layer_size):
            name = 'enc_{:d}'.format(i + 1)
            block = ConvBNReLU(in_channel, out_channel, 3, 1, 2,
                               dilation=2)
            setattr(self, name, block)

        self.att = AttentionModule(out_channel,
                                   att_module=att_module)

        # Dilated convs
        for i in range(layer_size - 1, 3, -1):
            name = 'dec_{:d}'.format(i)
            block = ConvBNReLU(in_channel + in_channel, in_channel, 3, 1, 2,
                               dilation=2)
            setattr(self, name, block)

        # Up convs 3x
        self.dec_3 = UpConvBNReLU(1024, 512, 3, 1, 1)
        self.dec_2 = UpConvBNReLU(768, 256, 3, 1, 1)
        self.dec_1 = UpConvBNReLU(384, 64, 3, 1, 1)

    def forward(self, input, mask):
        h_dict = {}  # for the output of enc_N
        h_dict['h_0'] = input

        h_key_prev = 'h_0'
        for i in range(1, self.layer_size + 1):
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev])
            # print(f"{l_key}, out size: {h_dict[h_key].shape}")
            h_key_prev = h_key

        h = h_dict[h_key]
        for i in range(self.layer_size - 1, 0, -1):
            enc_h_key = 'h_{:d}'.format(i)
            dec_l_key = 'dec_{:d}'.format(i)
            h = torch.cat([h, h_dict[enc_h_key]], dim=1)
            # print(f"{dec_l_key}, in size: {h.shape}")
            h = getattr(self, dec_l_key)(h)
            if i == 3:
                h = self.att(h, mask)

        return h

###############################################################################
# Models
###############################################################################


class VGG16FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class RFRNetv2(nn.Module):

    def __init__(self, base_channels=64, rfr_module_layer_size=6, recurrence=6):
        super(RFRNetv2, self).__init__()

        self.base_channels = base_channels
        self.recurrence = recurrence

        c = base_channels
        self.pconv1 = PConvBNReLU(3, c, 7, 2, 3)
        self.pconv2 = PConvBNReLU(c, c, 7, 1, 3)

        # Recursive
        self.pconv3 = PConvBNReLU(c, c, 7, 1, 3)
        self.pconv4 = PConvBNReLU(c, c, 7, 1, 3)
        self.rfr = RFRModule(c, rfr_module_layer_size)

        self.upconv1 = UpConvBNReLU(c, c, 3, 1, 1)

        self.tail1 = PConvBNReLU(c + 3, c//2, 3, 1, 1)
        self.tail2 = Bottleneck(c//2, c//8)

        self.out = nn.Conv2d(c, 3, 3, 1, 1, padding_mode="reflect")

    def forward(self, in_image, mask, recurrence=None):
        if recurrence is None:
            recurrence = self.recurrence

        x1, m1 = self.pconv1(in_image, mask)
        x1, m1 = self.pconv2(x1, m1)

        # Initialize
        x2, m2 = x1, m1
        # XXX Added this to be consistent with recurrence later
        x2 = x2 * m2
        feature_group = [x2.unsqueeze(2), ]
        mask_group = [m2.unsqueeze(2), ]
        # self.rfr.att.att.reset()

        for i in range(recurrence):

            x2, m2 = self.pconv3(x2, m2)
            x2, m2 = self.pconv4(x2, m2)

            x2 = self.rfr(x2, m2[:, 0:1, :, :])
            x2 = x2 * m2

            # print(f"recurrence #{i} - x2, m2: {x2.shape}, {m2.shape}")

            feature_group.append(x2.unsqueeze(2))
            mask_group.append(m2.unsqueeze(2))

        # dim=2 = time dimension
        x3 = torch.cat(feature_group, dim=2)
        m3 = torch.cat(mask_group, dim=2)

        # XXX Not sure what is this doing
        # amp_vec = m3.mean(dim=2)
        # x3 = (x3 * m3).mean(dim=2)/(amp_vec + epsilon)

        # Feature aggregation
        # Non-zero mean in time dimension; epsilon to avoid division by zero
        x3 = (x3 * m3).sum(dim=2)/(m3.sum(dim=2) + epsilon)
        # Last mask from all the recurrence masks
        m3 = m3[:, :, -1, :, :]

        x4 = self.upconv1(x3)
        # Use nearest interpolation for mask
        m4 = F.interpolate(m3, mask.shape[-2:],
                           mode="nearest")

        # Skip connections
        x5 = torch.cat([in_image, x4], dim=1)
        m5 = torch.cat([mask, m4], dim=1)
        x5, m5 = self.tail1(x5, m5)

        x6 = self.tail2(x5)
        x6 = torch.cat([x5, x6], dim=1)

        output = self.out(x6)
        return output, None

    def train(self, mode=True, finetune=False):
        super().train(mode)
        if finetune:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()


class RFRNetv3(nn.Module):

    def __init__(self, base_channels=64, rfr_module_layer_size=6, recurrence=6):
        super(RFRNetv3, self).__init__()

        self.base_channels = base_channels
        self.recurrence = recurrence

        c = base_channels
        self.pconv1 = PConvBNReLU(3, c, 7, 2, 3, multi_channel=False)
        self.pconv2 = PConvBNReLU(c, c, 7, 1, 3, multi_channel=False)

        # Recursive
        self.pconv3 = PConvBNReLU(c, c, 7, 1, 3, multi_channel=False)
        self.pconv4 = PConvBNReLU(c, c, 7, 1, 3, multi_channel=False)
        self.rfr = RFRModule(c, rfr_module_layer_size,
                             att_module=NONLocalBlock2D)

        # XXX Without padding in temporal dimension means recurrence should be at least 5
        self.featmerge = nn.Sequential(
            nn.Conv3d(c + 1, c, (3, 1, 1), (1, 1, 1), (0, 0, 0), bias=False),
            nn.BatchNorm3d(c),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(c, c, (3, 1, 1), (1, 1, 1), (0, 0, 0), bias=False),
            nn.BatchNorm3d(c),
            nn.LeakyReLU(inplace=True),

            # Pool over temporal dimension
            nn.AdaptiveAvgPool3d((1, None, None))
        )

        self.upconv1 = UpConvBNReLU(c, c, 3, 1, 1)

        self.tail1 = PConvBNReLU(c + 3, c//2, 3, 1, 1, multi_channel=False)
        self.tail2 = Bottleneck(c//2, c//8)

        self.out = nn.Conv2d(c, 3, 3, 1, 1, padding_mode="reflect")

    def forward(self, in_image, mask, recurrence=None):
        if recurrence is None:
            recurrence = self.recurrence

        x1, m1 = self.pconv1(in_image, mask[:, [0], :, :])
        # print(f"m1.shape: {m1.shape}")
        x1, m1 = self.pconv2(x1, m1)

        # Initialize
        x2, m2 = x1, m1
        # XXX Added this to be consistent with recurrence later
        x2 = x2 * m2
        feature_group = [x2.unsqueeze(2), ]
        mask_group = [m2.unsqueeze(2), ]

        for i in range(recurrence):
            x2, m2 = self.pconv3(x2, m2)
            x2, m2 = self.pconv4(x2, m2)

            x2 = self.rfr(x2, m2[:, [0], :, :])

            x2 = x2 * m2
            feature_group.append(x2.unsqueeze(2))
            mask_group.append(m2.unsqueeze(2))

        # dim=2 = time dimension
        x3 = torch.cat(feature_group, dim=2)
        m3 = torch.cat(mask_group, dim=2)

        # Feature aggregation along temporal dimension
        x3_cat_m3 = torch.cat([x3, m3], dim=1)
        x3 = self.featmerge(x3_cat_m3)
        assert x3.ndim == 5
        assert x3.size(2) == 1
        x3 = x3.squeeze(2)
        m3 = m3[:, :, -1, :, :]

        x4 = self.upconv1(x3)
        # Use nearest interpolation for mask
        m4 = F.interpolate(m3, x4.shape[-2:],
                           mode="nearest")

        # Skip connections
        x5 = torch.cat([in_image, x4], dim=1)
        m5 = m4

        x5, m5 = self.tail1(x5, m5)

        x6 = self.tail2(x5)
        x6 = torch.cat([x5, x6], dim=1)

        output = self.out(x6)
        return output, None

    def train(self, mode=True, finetune=False):
        super().train(mode)
        if finetune:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()


class RFRNetv5(nn.Module):

    def __init__(self, base_channels=64, rfr_module_layer_size=6, recurrence=6):
        super(RFRNetv5, self).__init__()

        self.base_channels = base_channels
        self.recurrence = recurrence

        c = base_channels
        self.pconv1 = PConvBNReLU(3, c, 7, 2, 3, multi_channel=False)
        self.pconv2 = PConvBNReLU(c, c, 7, 1, 3, multi_channel=False)

        # Recursive
        self.pconv3 = PConvBNReLU(c, c, 7, 1, 3, multi_channel=False)
        self.pconv4 = PConvBNReLU(c, c, 7, 1, 3, multi_channel=False)
        self.rfr = RFRModule(c, rfr_module_layer_size,
                             att_module=KnowledgeConsistentNONLocalBlock2D)

        # XXX Without padding in temporal dimension means recurrence should be at least 5
        self.featmerge = nn.Sequential(
            nn.Conv3d(c + 1, c, (3, 3, 3), (1, 1, 1), (0, 1, 1), bias=False),
            nn.BatchNorm3d(c),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(c, c, (3, 3, 3), (1, 1, 1), (0, 1, 1), bias=False),
            nn.BatchNorm3d(c),
            nn.LeakyReLU(inplace=True),

            # Pool over temporal dimension
            nn.AdaptiveAvgPool3d((1, None, None))
        )

        self.upconv1 = UpConvBNReLU(c, c, 3, 1, 1)

        self.tail1 = PConvBNReLU(c + 3, c//2, 3, 1, 1, multi_channel=False)
        self.tail2 = Bottleneck(c//2, c//8)

        self.out = nn.Conv2d(c, 3, 3, 1, 1, padding_mode="reflect")

    def forward(self, in_image, mask, recurrence=None, fp16=False):
        with autocast(fp16):
            if recurrence is None:
                recurrence = self.recurrence

            print(
                f"lamb, lamb_smooth: {torch.sigmoid(self.rfr.att.att.lamb).item()}, {torch.sigmoid(self.rfr.att.att.lamb_smooth).item()}")

            x1, m1 = self.pconv1(in_image, mask[:, [0], :, :])

            # print(f"m1.shape: {m1.shape}")
            x1, m1 = self.pconv2(x1, m1)

            # Initialize
            x2, m2 = x1, m1
            # XXX Added this to be consistent with recurrence later

            x2 = x2 * m2
            feature_group = [x2.unsqueeze(2), ]
            mask_group = [m2.unsqueeze(2), ]
            self.rfr.att.initialize_knowledge()

            for i in range(recurrence):
                x2, m2 = self.pconv3(x2, m2)
                x2, m2 = self.pconv4(x2, m2)
                x2 = self.rfr(x2, m2[:, [0], :, :])

                x2 = x2 * m2
                feature_group.append(x2.unsqueeze(2))
                mask_group.append(m2.unsqueeze(2))

            # dim=2 = time dimension
            x3 = torch.cat(feature_group, dim=2)
            m3 = torch.cat(mask_group, dim=2)

            # Feature aggregation along temporal dimension
            x3_cat_m3 = torch.cat([x3, m3], dim=1)
            x3 = self.featmerge(x3_cat_m3)
            assert x3.ndim == 5
            assert x3.size(2) == 1
            x3 = x3.squeeze(2)
            m3 = m3[:, :, -1, :, :]

            x4 = self.upconv1(x3)
            # Use nearest interpolation for mask
            m4 = F.interpolate(m3, x4.shape[-2:],
                               mode="nearest")

            # Skip connections
            x5 = torch.cat([in_image, x4], dim=1)
            m5 = m4

            x5, m5 = self.tail1(x5, m5)

            x6 = self.tail2(x5)
            x6 = torch.cat([x5, x6], dim=1)

            output = self.out(x6)
            return output, None

    def train(self, mode=True, finetune=False):
        super().train(mode)
        if finetune:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()


class RFRNetv6(nn.Module):

    def __init__(self, base_channels=64, rfr_module_layer_size=6, recurrence=6):
        super(RFRNetv6, self).__init__()

        self.base_channels = base_channels
        self.recurrence = recurrence

        c = base_channels
        self.pconv1 = PConvBNReLU(3, c, 7, 2, 3, multi_channel=False)
        self.pconv2 = PConvBNReLU(c, c, 7, 1, 3, multi_channel=False)

        # Recursive
        self.pconv3 = PConvBNReLU(c, c, 7, 1, 3, multi_channel=False)
        self.pconv4 = PConvBNReLU(c, c, 7, 1, 3, multi_channel=False)
        self.rfr = RFRModule(c, rfr_module_layer_size,
                             att_module=KnowledgeConsistentNONLocalBlock2D)

        # XXX Without padding in temporal dimension means recurrence should be at least 5
        self.featmerge = nn.Sequential(
            nn.Conv3d(c + 1, c, (3, 1, 1), (1, 1, 1), (0, 0, 0), bias=False),
            nn.BatchNorm3d(c),
            nn.LeakyReLU(inplace=True),

            # nn.Conv3d(c, c, (3, 1, 1), (1, 1, 1), (0, 0, 0), bias=False),
            # nn.BatchNorm3d(c),
            # nn.LeakyReLU(inplace=True),

            # Pool over temporal dimension
            nn.AdaptiveAvgPool3d((1, None, None))
        )

        self.upconv1 = UpConvBNReLU(c, c, 3, 1, 1)

        self.tail1 = PConvBNReLU(c + 3, c//2, 3, 1, 1, multi_channel=False)
        self.tail2 = Bottleneck(c//2, c//8)

        self.out = nn.Conv2d(c, 3, 3, 1, 1, padding_mode="reflect")

    def forward(self, in_image, mask, recurrence=None, fp16=False):
        with autocast(fp16):
            if recurrence is None:
                recurrence = self.recurrence

            # print(
            #     f"lamb, lamb_smooth: {torch.sigmoid(self.rfr.att.att.lamb).item()}, {torch.sigmoid(self.rfr.att.att.lamb_smooth).item()}")

            x1, m1 = self.pconv1(in_image, mask[:, [0], :, :])

            # print(f"m1.shape: {m1.shape}")
            x1, m1 = self.pconv2(x1, m1)

            # Initialize
            x2, m2 = x1, m1
            # XXX Added this to be consistent with recurrence later

            x2 = x2 * m2
            feature_group = [x2.unsqueeze(2), ]
            mask_group = [m2.unsqueeze(2), ]
            self.rfr.att.initialize_knowledge()

            for i in range(recurrence):
                x2, m2 = self.pconv3(x2, m2)
                x2, m2 = self.pconv4(x2, m2)
                x2 = self.rfr(x2, m2[:, [0], :, :])

                x2 = x2 * m2
                feature_group.append(x2.unsqueeze(2))
                mask_group.append(m2.unsqueeze(2))

            # dim=2 = time dimension
            feature_group = torch.cat(feature_group, dim=2)
            mask_group = torch.cat(mask_group, dim=2)

            # Feature aggregation along temporal dimension
            x3_cat_m3 = torch.cat([feature_group, mask_group], dim=1)
            x3 = self.featmerge(x3_cat_m3)
            assert x3.ndim == 5
            assert x3.size(2) == 1
            x3 = x3.squeeze(2)
            m3 = mask_group[:, :, -1, :, :]

            feature_group = torch.cat([feature_group, x3.unsqueeze(2)], dim=2)
            mask_group = torch.cat([mask_group, m3.unsqueeze(2)], dim=2)

            reconstructions = []
            processed_masks = []
            for t in range(feature_group.size(2)):
                _x = feature_group[:, :, t, :, :]
                _m = mask_group[:, :, t, :, :]

                x4 = self.upconv1(_x)
                # Use nearest interpolation for mask
                m4 = F.interpolate(_m, x4.shape[-2:],
                                   mode="nearest")

                # Skip connections
                x5 = torch.cat([in_image, x4], dim=1)
                m5 = m4

                x5, m5 = self.tail1(x5, m5)

                x6 = self.tail2(x5)
                x6 = torch.cat([x5, x6], dim=1)

                output = self.out(x6)

                reconstructions.append(output)
                processed_masks.append(m5)

            # reconstructions.reverse()
            # processed_masks.reverse()

            return output, (reconstructions, processed_masks)

    def train(self, mode=True, finetune=False):
        super().train(mode)
        if finetune:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()


if __name__ == "__main__":
    from torchsummary import summary
    model = RFRNetv6()
    inputs = (torch.rand(2, 3, 256, 256), torch.rand(2, 3, 256, 256))
    # summary(model, inputs)
    out = model(*inputs)
    print(len(out[1][0]))
    print(out[1][0][-1].shape)
    print(out[1][1][-1].shape)
