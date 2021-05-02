import math
import pdb
import os

import torch
from torch import nn
from torch.nn import functional as F
import onnxruntime
import time
import torchvision.utils as utils
from PIL import Image
import torchvision.transforms as transforms

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return F.leaky_relu(input + bias.view(1, bias.shape[0], 1, 1), negative_slope=0.2) * scale

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        b, c, h, w = input.shape
        input = input.view(b * c, 1, h, w)
        weight = self.kernel.view(1, 1, 4, 4)
        out = F.conv2d(input, weight, padding = self.pad)
        h, w = out.shape[2], out.shape[3]
        out = out.view(b, c, h, w)
        return out


class EqualLinear(torch.jit.ScriptModule):
    def __init__(self, in_dim, out_dim, bias_init=0, lr_mul=1):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out

class EqualLinearWithLeakyRelu(nn.Module):
    '''Add this class for onnx -- data driven flow is difficult tracing.'''
    def __init__(self, in_dim, out_dim, bias_init=0, lr_mul=1):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def leaky_relu(self, input, bias, negative_slope=0.2, scale=2 ** 0.5):
        return F.leaky_relu(input + bias.view(1, bias.shape[0]), negative_slope=0.2) * scale

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale)
        out = self.leaky_relu(out, self.bias * self.lr_mul)

        return out

class ModulatedConv2d(torch.jit.ScriptModule):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        w_space_dim,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(w_space_dim, in_channel, bias_init=1)

    # @torch.jit.script_method
    def forward(self, input, style):
        batch, in_channel, height, width = input.shape[0], input.shape[1], input.shape[2], input.shape[3]

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style
        # Norm weight !!!
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8).view(batch, self.out_channel, 1, 1, 1)
        weight = weight * demod
        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        input = input.view(1, batch * in_channel, height, width)
        # input.size() -- torch.Size([1, 512, 4, 4])
        # weight.size() -- torch.Size([512, 512, 3, 3])
        # TracerWarning: Converting a tensor to a Python integer might cause the trace 
        # to be incorrect
        # out = F.conv2d(input, weight, padding=self.kernel_size//2, groups=batch)
        out = F.conv2d(input, weight, padding=1, groups=1)

        # print("{} -- {} --> {}".format(input.size(), weight.size(), out.size()))
        # torch.Size([1, 4608, 4, 4]) -- torch.Size([4608, 512, 3, 3]) --> torch.Size([1, 4608, 4, 4])
        # torch.Size([1, 4608, 8, 8]) -- torch.Size([4608, 512, 3, 3]) --> torch.Size([1, 4608, 8, 8])
        # torch.Size([1, 4608, 16, 16]) -- torch.Size([4608, 512, 3, 3]) --> torch.Size([1, 4608, 16, 16])
        # torch.Size([1, 4608, 32, 32]) -- torch.Size([4608, 512, 3, 3]) --> torch.Size([1, 4608, 32, 32])
        # torch.Size([1, 4608, 64, 64]) -- torch.Size([4608, 512, 3, 3]) --> torch.Size([1, 4608, 64, 64])
        # torch.Size([1, 2304, 128, 128]) -- torch.Size([2304, 256, 3, 3]) --> torch.Size([1, 2304, 128, 128])
        # torch.Size([1, 1152, 256, 256]) -- torch.Size([1152, 128, 3, 3]) --> torch.Size([1, 1152, 256, 256])
        # torch.Size([1, 576, 512, 512]) -- torch.Size([576, 64, 3, 3]) --> torch.Size([1, 576, 512, 512])
        # torch.Size([1, 288, 1024, 1024]) -- torch.Size([288, 32, 3, 3]) --> torch.Size([1, 288, 1024, 1024])

        height, width = out.shape[2], out.shape[3]
        return out.view(batch, self.out_channel, height, width)


class ModulatedConv2dWithUpsample(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        w_space_dim,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        factor = 2
        p = (len(blur_kernel) - factor) - (kernel_size - 1)
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2 + 1
        self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(w_space_dim, in_channel, bias_init=1)


    def forward(self, input, style):
        batch, in_channel, height, width = input.shape[0], input.shape[1], input.shape[2], input.shape[3]

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        # Norm weight !!!
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        input = input.view(1, batch * in_channel, height, width)
        weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
        # out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
        out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=1)
        height, width = out.shape[2], out.shape[3]
        out = out.view(batch, self.out_channel, height, width)
        out = self.blur(out)
        # print("{} -- {} --> {}".format(input.size(), weight.size(), out.size()))
        # torch.Size([1, 4608, 4, 4]) -- torch.Size([4608, 512, 3, 3]) --> torch.Size([9, 512, 8, 8])
        # torch.Size([1, 4608, 8, 8]) -- torch.Size([4608, 512, 3, 3]) --> torch.Size([9, 512, 16, 16])
        # torch.Size([1, 4608, 16, 16]) -- torch.Size([4608, 512, 3, 3]) --> torch.Size([9, 512, 32, 32])
        # torch.Size([1, 4608, 32, 32]) -- torch.Size([4608, 512, 3, 3]) --> torch.Size([9, 512, 64, 64])
        # torch.Size([1, 4608, 64, 64]) -- torch.Size([4608, 256, 3, 3]) --> torch.Size([9, 256, 128, 128])
        # torch.Size([1, 2304, 128, 128]) -- torch.Size([2304, 128, 3, 3]) --> torch.Size([9, 128, 256, 256])
        # torch.Size([1, 1152, 256, 256]) -- torch.Size([1152, 64, 3, 3]) --> torch.Size([9, 64, 512, 512])
        # torch.Size([1, 576, 512, 512]) -- torch.Size([576, 32, 3, 3]) --> torch.Size([9, 32, 1024, 1024])
        return out

class ModulatedConv2dWithoutNormWeight(nn.Module):
    def __init__(
        self,
        in_channel = 3,
        out_channel = 1,
        kernel_size = 1,
        w_space_dim = 512
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(w_space_dim, in_channel, bias_init=1)

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape[0], input.shape[1], input.shape[2], input.shape[3]
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style
        # (Pdb) self.weight.size(), style.size()
        # (torch.Size([1, 3, 512, 1, 1]), torch.Size([9, 1, 512, 1, 1]))

        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)
        # batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size -- (27, 512, 1, 1)

        input = input.view(1, batch * in_channel, height, width)
        # TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect
        # out = F.conv2d(input, weight, padding=self.kernel_size//2, groups=batch)
        out = F.conv2d(input, weight, padding=0, groups=1)
        # print("{} -- {} --> {}".format(input.size(), weight.size(), out.size()))
        # torch.Size([1, 4608, 8, 8]) -- torch.Size([27, 512, 1, 1]) --> torch.Size([1, 27, 8, 8])
        # torch.Size([1, 4608, 16, 16]) -- torch.Size([27, 512, 1, 1]) --> torch.Size([1, 27, 16, 16])
        # torch.Size([1, 4608, 32, 32]) -- torch.Size([27, 512, 1, 1]) --> torch.Size([1, 27, 32, 32])
        # torch.Size([1, 4608, 64, 64]) -- torch.Size([27, 512, 1, 1]) --> torch.Size([1, 27, 64, 64])
        # torch.Size([1, 2304, 128, 128]) -- torch.Size([27, 256, 1, 1]) --> torch.Size([1, 27, 128, 128])
        # torch.Size([1, 1152, 256, 256]) -- torch.Size([27, 128, 1, 1]) --> torch.Size([1, 27, 256, 256])
        # torch.Size([1, 576, 512, 512]) -- torch.Size([27, 64, 1, 1]) --> torch.Size([1, 27, 512, 512])
        # torch.Size([1, 288, 1024, 1024]) -- torch.Size([27, 32, 1, 1]) --> torch.Size([1, 27, 1024, 1024])

        height, width = out.shape[2], out.shape[3]
        out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))        # learn by training ...

    def forward(self, image):
        # batch, _, height, width = image.shape
        # noise = image.new_empty(batch, 1, height, width).normal_()
        # Onnx does not support new_empty and normal_, so we develop it
        noise = torch.randn_like(image)
        return image + self.weight * noise * 0.90 # reduce noise ?


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))
        # (Pdb) self.input.size()
        # torch.Size([1, 512, 4, 4])

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        # out.size() -- torch.Size([9, 512, 4, 4])
        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        w_space_dim,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, w_space_dim,
                                    blur_kernel=blur_kernel
                                    )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style):
        out = self.conv(input, style)
        out = self.noise(out)
        out = self.activate(out)

        return out

class StyledConvWithUpsample(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        w_space_dim,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.conv = ModulatedConv2dWithUpsample(in_channel, out_channel, kernel_size, w_space_dim,
                                    blur_kernel=blur_kernel
                                    )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style):
        out = self.conv(input, style)
        out = self.noise(out)
        out = self.activate(out)

        return out

class ToRGB(nn.Module):
    ''' to_rgbs ...'''
    def __init__(self, in_channel, w_space_dim):
        super().__init__()
        self.conv = ModulatedConv2dWithoutNormWeight(in_channel, out_channel=3, kernel_size=1, w_space_dim=w_space_dim)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        return self.conv(input, style) + self.bias


class ToRGBWithUpsample(nn.Module):
    ''' to_rgbs ...'''
    def __init__(self, in_channel, w_space_dim):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)

        self.conv = ModulatedConv2dWithoutNormWeight(in_channel, out_channel=3, kernel_size=1, w_space_dim=w_space_dim)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip):
        out = self.conv(input, style) + self.bias
        skip = self.upsample(skip)
        return out + skip


class StyleGAN2Transformer(nn.Module):
    def __init__(self, w_space_dim=512, n_mlp=8, lr_mlp=0.01):
        super().__init__()

        layers = [PixelNorm()]
        for i in range(n_mlp):
            # bias=True, bias_init=0, lr_mul=1, activation=None
            layers.append(
                EqualLinearWithLeakyRelu(w_space_dim, w_space_dim, lr_mul=lr_mlp)
            )

        self.style = nn.Sequential(*layers)
        self.w_space_dim = w_space_dim
        self.mean_wcode = torch.randn(4096, self.w_space_dim).mean(0, keepdim=True)

    def forward(self, zcode):
        '''Transform zcode to wcode. zcode format Bx1x1x512, return wcode: Bx1x1x512'''
        simple_zcode = zcode.squeeze(1).squeeze(1)
        # [9, 1, 1, 512] --> [9, 512]
        simple_wcode = self.style(simple_zcode)

        temp_mean_wcode = self.style(self.mean_wcode.to(simple_wcode.device))

        # Truncation for reducing strange faces ...
        simple_wcode = 0.75*simple_wcode + 0.25 * temp_mean_wcode

        wcode = simple_wcode.unsqueeze(1).unsqueeze(1)

        return wcode


class Generator(nn.Module):
    def __init__(
        self,
        resolution=1024,
        w_space_dim=512,
        n_mlp=8,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.resolution = resolution

        self.w_space_dim = w_space_dim
        self.n_mlp = n_mlp
        self.channel_multiplier = channel_multiplier

        self.log_size = int(math.log(resolution, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latents = self.log_size * 2 - 2

        self.style = StyleGAN2Transformer(w_space_dim, n_mlp, lr_mlp).style

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, w_space_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(self.channels[4], w_space_dim)


        convs_list = []
        # self.convs = nn.ModuleList()

        to_rgbs_list = []
        # self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(
                f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            convs_list.append(
                StyledConvWithUpsample(in_channel, out_channel, 3, w_space_dim, blur_kernel=blur_kernel)
            )

            convs_list.append(
                StyledConv(out_channel, out_channel, 3, w_space_dim, blur_kernel=blur_kernel)
            )
            # self.to_rgbs.append(ToRGBWithUpsample(out_channel, w_space_dim))
            to_rgbs_list.append(ToRGBWithUpsample(out_channel, w_space_dim))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs_list)
        self.to_rgbs = nn.Sequential(*to_rgbs_list)

        # self.eigvectors = torch.zeros(w_space_dim, w_space_dim)

    def forward(self, wcode):
        # wcode = self.style(zcode)
        # self.num_latents -- 18

        latent = wcode.squeeze(1).repeat(1, self.num_latents, 1)
        # (Pdb) latent.size() -- torch.Size([1, 18, 512])

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        # [::2] -- start 0, step 2, --> 0, 2, 4, 6, 8 ...
        # [1::2] -- start 1, step 2, --> 1, 3, 5, 7, 9 ...
        # # https://github.com/prokotg/colorization/blob/master/colorizers/siggraph17.py
        # for conv1, conv2, noise1, noise2, to_rgb in zip(
        #     self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        # ):
        #     out = conv1(out, latent[:, i], noise=noise1)
        #     out = conv2(out, latent[:, i + 1], noise=noise2)
        #     skip = to_rgb(out, latent[:, i + 2], skip)

        #     i += 2

        # loop count 8 times
        # i = 1, 3, 5, 7, 9, 11, 13, 15
        out = self.convs[0](out, latent[:, 1])
        out = self.convs[1](out, latent[:, 2])
        skip = self.to_rgbs[0](out, latent[:, 3], skip)

        out = self.convs[2](out, latent[:, 3])
        out = self.convs[3](out, latent[:, 4])
        skip = self.to_rgbs[1](out, latent[:, 5], skip)

        out = self.convs[4](out, latent[:, 5])
        out = self.convs[5](out, latent[:, 6])
        skip = self.to_rgbs[2](out, latent[:, 7], skip)

        out = self.convs[6](out, latent[:, 7])
        out = self.convs[7](out, latent[:, 8])
        skip = self.to_rgbs[3](out, latent[:, 9], skip)

        out = self.convs[8](out, latent[:, 9])
        out = self.convs[9](out, latent[:, 10])
        skip = self.to_rgbs[4](out, latent[:, 11], skip)

        out = self.convs[10](out, latent[:, 11])
        out = self.convs[11](out, latent[:, 12])
        skip = self.to_rgbs[5](out, latent[:, 13], skip)

        out = self.convs[12](out, latent[:, 13])
        out = self.convs[13](out, latent[:, 14])
        skip = self.to_rgbs[6](out, latent[:, 15], skip)

        out = self.convs[14](out, latent[:, 15])
        out = self.convs[15](out, latent[:, 16])
        skip = self.to_rgbs[7](out, latent[:, 17], skip)

        # image = skip
        '''Post image, from [-1.0, 1.0] to [0.0, 1.0].'''
        image = ((skip + 1.0)/2.0).clamp(0.0, 1.0)

        return image

    # def eigen(self, index):
    #     # eigen vector for dim index ...
    #     assert index < self.w_space_dim
    #     return self.eigvectors[:, index]


def get_decoder():
    ''' Get generator'''

    # resolution, w_space_dim, n_mlp
    print("Creating decoder ...")
    model = Generator(1024, 512, 8)
    checkpoint = "models/ImageGanDecoder.pth"

    if not os.path.exists(checkpoint):
        print("Model '{}' does not exist.".format(checkpoint))
    else:
        source_state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        source_state_dict = source_state_dict["g_ema"]
        target_state_dict = model.state_dict()
        for n in target_state_dict.keys():
            p = source_state_dict[n]
            target_state_dict[n].copy_(p)
    return model

    # Start weight factorizing
    if False:
        print("Factorizing decoder weights ...")
        modulate = {
            k: v
            for k, v in model_weights.items()
            if "modulation" in k and "to_rgbs" not in k and "weight" in k
        }
        # (Pdb) modulate.keys()
        # dict_keys(['conv1.conv.modulation.weight',
        #     'to_rgb1.conv.modulation.weight',
        #     'convs.0.conv.modulation.weight',
        #     ......
        #     'convs.15.conv.modulation.weight'])
        weight_mat = []
        for k, v in modulate.items():
            weight_mat.append(v)
        W = torch.cat(weight_mat, 0)
        # torch.svd(W).S, torch.svd(W).V ...
        model.eigvectors = torch.svd(W).V

        # (Pdb) eigvectors.size()
        # torch.Size([512, 512])

    return model

def get_transformer():
    ''' Get transformer'''

    # resolution, w_space_dim, n_mlp
    print("Creating transformer weight file ...")
    checkpoint = "models/ImageGanTransformer.pth"

    model = StyleGAN2Transformer()
    if not os.path.exists(checkpoint):
        # Create 
        decoder = get_decoder()
        source_state_dict = decoder.style.state_dict()
        target_state_dict = model.state_dict()
        for n, p in source_state_dict.items():
            tn = "style." + n
            if tn in target_state_dict.keys():
                target_state_dict[tn].copy_(p)
            else:
                raise KeyError(tn)
        torch.save(model.state_dict(), checkpoint)

    model_weights = torch.load(checkpoint)
    model.load_state_dict(model_weights)

    return model


def export_onnx():
    """Export onnx model."""

    import numpy as np
    import onnx
    import onnxruntime
    from onnx import optimizer

    # ------- For Decoder -----------------------
    # KNOWN ISSUE: NOT SUPPORT dynamic_axes, NOT CUDA inference For this model !!!
    onnx_file_name = "output/image_gandecoder.onnx"
    # dummy_input = torch.randn(1, 1, 1, 512).cuda()
    dummy_input = torch.randn(1, 1, 1, 512)

    # 1. Create and load model.
    torch_model = get_decoder()
    # torch_model = torch_model.cuda()
    torch_model.eval()

    # 2. Model export
    print("Export decoder model ...")

    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(torch_model, dummy_input, onnx_file_name,
                      input_names=input_names,
                      output_names=output_names,
                      verbose=True,
                      opset_version=11,
                      keep_initializers_as_inputs=False,
                      export_params=True)

    # 3. Optimize model
    print('Checking model ...')
    onnx_model = onnx.load(onnx_file_name)
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model.graph)
    # https://github.com/onnx/optimizer

    # 4. Visual model
    # python -c "import netron; netron.start('output/image_gandecoder.onnx')"


    # ------- For Transformer -----------------------
    onnx_file_name = "output/image_gantransformer.onnx"
    # dummy_input = torch.randn(1, 1, 1, 512).cuda()
    dummy_input = torch.randn(1, 1, 1, 512)

    # 1. Create and load model.
    torch_model = get_transformer()
    # torch_model = torch_model.cuda()
    torch_model.eval()

    # 2. Model export
    print("Export transformer model ...")

    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {'input': {0: "batch"}, 'output': {0: "batch"}}
    torch.onnx.export(torch_model, dummy_input, onnx_file_name,
                      input_names=input_names,
                      output_names=output_names,
                      verbose=True,
                      opset_version=11,
                      keep_initializers_as_inputs=False,
                      export_params=True,
                      dynamic_axes=dynamic_axes)


    # 3. Optimize model
    print('Checking model ...')
    onnx_model = onnx.load(onnx_file_name)
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model.graph)
    # https://github.com/onnx/optimizer

    # 4. Visual model
    # python -c "import netron; netron.start('output/image_gantransformer.onnx')"

def verify_onnx():
    """Verify onnx model."""

    import numpy as np
    import onnxruntime

    # ------- For Transformer -----------------------
    onnx_file_name = "output/image_gantransformer.onnx"
    torch_model = get_transformer()
    torch_model.eval()
    onnxruntime_engine = onnx_model_load(onnx_file_name)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    dummy_input = torch.randn(1, 1, 1, 512)
    with torch.no_grad():
        torch_output = torch_model(dummy_input)

    onnxruntime_inputs = {
        onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_input)}

    onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)
    np.testing.assert_allclose(
        to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-02, atol=1e-02)
    print("Transformer onnx model has been tested with ONNXRuntime, the result sounds good !")

    # ------- For Decoder -----------------------
    onnx_file_name = "output/image_gandecoder.onnx"
    torch_model = get_decoder()
    torch_model.eval()
    onnxruntime_engine = onnx_model_load(onnx_file_name)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    dummy_input = torch.randn(1, 1, 1, 512)
    with torch.no_grad():
        torch_output = torch_model(dummy_input)
    onnxruntime_inputs = {
        onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_input)}
    onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)
    np.testing.assert_allclose(
        to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-02, atol=1e-02)
    print("Decoder onnx model has been tested with ONNXRuntime, the result sounds good !")


def export_torch():
    """Export torch model."""

    print("================> Torch Script Tansformer ...")
    # ------- For Transformer -----------------------
    script_file = "output/image_gantransformer.pt"

    # 1. Load model
    model = get_transformer()
    model.eval()

    # 2. Model export
    dummy_input = torch.randn(1, 1, 1, 512)
    traced_script_module = torch.jit.trace(
        model, dummy_input, _force_outplace=True)
    traced_script_module.save(script_file)


    print("================> Torch Script Decoder ...")

    # ------- For Decoder -----------------------
    script_file = "output/image_gandecoder.pt"

    # 1. Load model
    print("Loading model ...")
    model = get_decoder()
    model.eval()

    # 2. Model export
    print("Export model ...")
    dummy_input = torch.randn(1, 1, 1, 512)
    traced_script_module = torch.jit.trace(
        model, dummy_input, _force_outplace=True)
    traced_script_module.save(script_file)


def grid_image(tensor, nrow=3):
    grid = utils.make_grid(tensor, nrow=nrow)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
        1, 2, 0).to('cpu', torch.uint8).numpy()
    image = Image.fromarray(ndarr)
    return image

def python_sample(number):
    '''Sample.'''
    from model import model_setenv, model_device

    # Random must be set before torch model, it is realy strange !!! ...
    zcodes = torch.randn(number, 1, 1, 512)

    model_setenv()
    device = model_device()
    decoder = get_decoder()
    decoder = decoder.to(device)
    decoder.eval()

    transformer = get_transformer()
    transformer = transformer.to(device)
    transformer.eval()

    print("Generating samples ...")
    start_time = time.time()

    images = []
    for i in range(number):
        zcode = zcodes[i:i+1, :, :, :]
        zcode = zcode.to(device)
        with torch.no_grad():
            wcode = transformer(zcode)
            image = decoder(wcode)
        images.append(image)

    spend_time = time.time() - start_time
    print("Spend time: {:.2f} seconds".format(spend_time))

    nrow = int(math.sqrt(number) + 0.5) 
    image = grid_image(torch.cat(images, dim=0), nrow=nrow)
    image.show()
    image.save("output/sample-9.png")


def onnx_model_load(onnx_file):
    sess_options = onnxruntime.SessionOptions()
    # sess_options.log_severity_level = 0

    # Set graph optimization level
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    onnx_model = onnxruntime.InferenceSession(onnx_file, sess_options)
    # onnx_model.set_providers(['CUDAExecutionProvider'])
    print("Onnx model engine: ", onnx_model.get_providers(), "Device: ", onnxruntime.get_device())

    return onnx_model

def onnx_model_forward(onnx_model, input):
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    onnxruntime_inputs = {onnx_model.get_inputs()[0].name: to_numpy(input)}
    onnxruntime_outputs = onnx_model.run(None, onnxruntime_inputs)
    return torch.from_numpy(onnxruntime_outputs[0])

def onnx_sample(number):
    # Random must be set before torch model, it is realy strange !!! ...
    zcodes = torch.randn(number, 1, 1, 512)

    decoder = onnx_model_load("output/image_gandecoder.onnx")
    transformer = onnx_model_load("output/image_gantransformer.onnx")

    print("Generating onnx samples ...")
    start_time = time.time()

#     bad = torch.Tensor([0.04,-0.05,-1.74,0.12,0.83,0.21,0.40,0.51,1.94,1.19,1.17,0.43,-0.85,2.37,-0.71,-1.39,-1.10,0.07,0.34,0.00,0.31,0.37,-2.07,0.00,-0.60,-0.96,-0.38,-0.35,-0.06,0.13,-1.68,0.57,-0.11,-1.36,-0.71,-0.53,0.24,0.27,-0.72,-1.61,-0.23,1.47,1.21,2.23,-1.90,0.32,-0.47,-1.28,-0.60,0.35,1.71,-0.62,0.52,-1.33,0.95,-0.66,0.13,-1.03,1.34,-0.68,1.91,-0.76,0.33,-0.53,-0.35,-0.92,-2.00,-0.27,0.30,0.20,1.01,0.03,-0.47,1.35,-0.12,-0.00,1.62,0.32,2.51,0.06,0.77,0.15,1.79,0.90,-0.24,0.05,1.15,0.11,-0.53,-0.44,2.07,-0.25,1.94,0.24,0.17,2.58,-1.56,-0.80,-0.56,-0.30,-0.96,0.01,1.74,1.04,-1.79,0.73,-0.43,-0.39,0.12,0.60,-1.54,-0.32,-0.47,-2.44,0.42,-0.92,0.86,-0.34,1.54,-1.07,-0.34,0.51,0.56,-1.87,0.35,0.03,-0.92,0.34,-1.09,-0.15,-1.02,-0.67,-1.02,-1.94,0.00,-0.39,-0.85,1.11,-0.18,1.44,-0.50,2.39,-1.88,-1.37,0.27,-1.22,-0.11,1.32,0.50,-1.62,1.01,1.23,0.36,0.64,-1.64,0.14,-1.22,0.22,-1.39,0.75,-1.70,-1.14,1.02,-0.83,-0.45,-1.44,2.15,-0.29,-1.17,-0.21,-0.23,0.12,0.92,1.04,-1.61,-1.90,2.14,-1.01,0.20,1.11,-0.45,-1.46,-0.65,1.23,0.28,-0.82,-0.16,1.20,-0.47,0.31,0.60,-0.37,1.82,-0.82,-0.96,-1.14,-0.09,0.08,2.12,1.09,-0.98,1.21,-0.70,-0.17,1.45,-0.50,-1.45,1.40,-0.37,1.19,-3.06,1.20,0.92,-1.83,0.72,0.70,-0.75,0.93,0.14,0.39,0.97,-0.36,0.88,-0.07,-0.43,-1.56,1.93,0.03,2.11,-0.57,-0.55,-0.25,1.20,0.19,1.24,-0.31,0.86,0.22,-0.46,-0.48,-0.36,1.01,-1.03,0.20,1.05,-0.45,1.03,1.38,1.72,0.41,-1.00,0.42,1.05,2.35,0.38,-0.25,-0.88,0.09,0.30,1.45,-1.63,-0.87,-0.39,-0.86,0.66,-0.55,-0.21,-0.94,1.12,0.13,1.10,1.13,0.70,-0.30,-1.46,1.50,1.02,0.11,-1.59,-0.53,0.64,-1.15,1.35,-0.98,-0.84,-0.72,-1.11,0.10,2.08,-0.37,2.06,-0.24,1.63,-0.73,-0.03,-1.04,0.31,-2.15,0.61,-1.13,-0.56,0.07,1.21,-2.38,-1.23,-0.30,0.50,-0.06,2.54,-0.04,-1.07,-1.12,1.17,0.11,0.03,0.26,-0.29,1.69,0.51,-0.84,-0.37,-0.32,-1.23,0.94,0.06,0.69,1.01,0.22,-0.10,-2.64,-0.74,0.33,-0.33,0.16,-1.12,-0.42,0.84,0.16,1.41,-0.13,1.86,0.03,0.37,0.98,0.24,0.51,0.13,-1.58,1.03,-0.30,1.10,-0.50,0.46,0.48,-0.25,0.08,0.35,-0.42,-0.42,0.82,1.17,-0.04,-0.36,0.51,0.04,-1.17,-2.64,1.25,-0.50,-0.33,1.33,0.71,1.87,-0.55,0.26,-1.42,1.73,-0.84,2.50,-0.13,1.43,-0.46,-0.62,1.01,0.55,-0.16,0.43,-0.35,1.49,-1.13,-0.31,-0.82,0.94,-0.06,-0.28,0.79,-0.33,-0.67,0.01,0.24,0.11,-0.82,-0.71,-0.62,-0.02,1.41,0.91,-0.92,-0.26,-0.13,0.36,-0.90,1.40,0.02,-2.10,-0.37,0.71,0.37,0.43,1.41,2.03,-0.26,-0.62,2.33,-1.22,0.96,0.52,0.42,-0.84,2.54,-0.21,0.09,0.28,-0.85,-0.93,0.27,-0.53,-0.33,0.35,-0.47,0.76,-0.10,0.03,0.15,0.22,-0.63,0.16,0.68,1.61,-0.35,0.72,-0.96,0.67,0.25,-0.03,0.26,-0.21,-2.23,0.78,0.19,0.96,-1.52,1.85,1.49,-1.41,-0.92,0.73,-0.97,0.98,-0.58,-1.44,-0.21,-0.43,1.24,-0.31,-1.02,-0.26,-0.43,-0.86,-1.37,0.70,-0.44,0.29,0.15,-1.03,-0.38,1.38,0.90,1.41,0.77,-1.23,-0.72,-1.19,0.53,-0.03,-0.28,0.42,0.15,-0.39,-2.07,-0.09,1.15,-0.16,1.77,0.28,-1.71,-0.82,-1.09,-0.43,1.21,0.68,-0.00]
# )
#     bad = bad.reshape(1, 1, 1, 512)

    images = []
    for i in range(number):
        zcode = zcodes[i:i+1, :, :, :]
        wcode = onnx_model_forward(transformer, zcode)
        image = onnx_model_forward(decoder, wcode)
        images.append(image)
    
    spend_time = time.time() - start_time
    print("Spend time: {:.2f} seconds".format(spend_time))

    nrow = int(math.sqrt(number) + 0.5) 
    image = grid_image(torch.cat(images, dim=0), nrow=nrow)
    image.show()
    image.save("output/sample-onnx-9.png")


if __name__ == '__main__':
    """Test Tools ..."""
    import argparse
    import os

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--export', help="Export onnx model", action='store_true')
    parser.add_argument(
        '--verify', help="Verify onnx model", action='store_true')
    parser.add_argument(
        '--sample', help="Sample 9 faces", action='store_true')

    parser.add_argument('--output', type=str,
                        default="output", help="output folder")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.export:
        export_torch()
        export_onnx()

    if args.verify:
        verify_onnx()

    if args.sample:
        onnx_sample(9)
        python_sample(9)
