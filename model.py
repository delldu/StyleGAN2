import math
import random
import os
from pathlib import Path

import lpips

import torch
from torch import nn
from torch.nn import functional as F

import torchvision.utils as utils
from PIL import Image

from torch import optim
from torchvision import transforms as T
from tqdm import tqdm

import pdb

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn2d_native(
        input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
    )
    return out


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )

    # xxxx8888
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    rest_dim = [1] * (input.ndim - bias.ndim - 1)
    return F.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2) * scale


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


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


# class Downsample(nn.Module):
#     def __init__(self, kernel, factor=2):
#         super().__init__()

#         self.factor = factor
#         kernel = make_kernel(kernel)
#         self.register_buffer("kernel", kernel)

#         p = kernel.shape[0] - factor

#         pad0 = (p + 1) // 2
#         pad1 = p // 2

#         self.pad = (pad0, pad1)

#     def forward(self, input):
#         out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

#         return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


# class EqualConv2d(nn.Module):
#     def __init__(
#         self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
#     ):
#         super().__init__()

#         self.weight = nn.Parameter(
#             torch.randn(out_channel, in_channel, kernel_size, kernel_size)
#         )
#         self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

#         self.stride = stride
#         self.padding = padding

#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_channel))
#         else:
#             self.bias = None

#     def forward(self, input):
#         out = F.conv2d(
#             input,
#             self.weight * self.scale,
#             bias=self.bias,
#             stride=self.stride,
#             padding=self.padding,
#         )

#         return out

#     def __repr__(self):
#         return (
#             f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
#             f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
#         )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        z_space_dim,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(z_space_dim, in_channel, bias_init=1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape[0], input.shape[1], input.shape[2], input.shape[3]

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            height, width = out.shape[2], out.shape[3]
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            height, width = input.shape[2], input.shape[3]
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            height, width = out.shape[2], out.shape[3]
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            # input.size() -- torch.Size([1, 512, 4, 4])
            # weight.size() -- torch.Size([512, 512, 3, 3])
            # self.padding -- 1
            # batch -- tensor(1)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            height, width = out.shape[2], out.shape[3]
            out = out.view(batch, self.out_channel, height, width)


        return out

class NoModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        z_space_dim
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

        self.modulation = EqualLinear(z_space_dim, in_channel, bias_init=1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape[0], input.shape[1], input.shape[2], input.shape[3]

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style
        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=self.padding, groups=batch)

        height, width = out.shape[2], out.shape[3]
        out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            # noise = image.new_empty(batch, 1, height, width).normal_()
            # Onnx does not support new_empty and normal_, so we develop it
            noise = torch.rand_like(image)
            mu = noise.mean()
            var = noise.std()
            noise = (noise - mu)/(var + 1e-6)
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        z_space_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, z_space_dim, 
            upsample=upsample,
            blur_kernel=blur_kernel
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, z_space_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = NoModulatedConv2d(in_channel, 3, 1, z_space_dim)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size = 1024,
        z_space_dim = 512,
        n_mlp = 8,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.z_space_dim = z_space_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            # bias=True, bias_init=0, lr_mul=1, activation=None
            layers.append(
                EqualLinear(z_space_dim, z_space_dim, lr_mul=lr_mlp, activation="fused_lrelu")
            )

        self.style = nn.Sequential(*layers)

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
        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, z_space_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(self.channels[4], z_space_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(in_channel, out_channel, 3, z_space_dim, upsample=True, blur_kernel=blur_kernel)
            )

            self.convs.append(
                StyledConv(out_channel, out_channel, 3, z_space_dim, blur_kernel=blur_kernel)
            )

            self.to_rgbs.append(ToRGB(out_channel, z_space_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

        self.last_latent = None

    def make_noise(self):
        device = self.input.input.device
        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]
        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
        # pdb.set_trace() -- len(noises) == 17
        # [4, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024]
        return noises

    def mean_latent(self, n_latent):
        zcode = torch.randn(
            n_latent, self.z_space_dim, device=self.input.input.device
        )
        latent = self.style(zcode).mean(0, keepdim=True)

        return latent

    def get_latent(self, zcode):
        return self.style(zcode)

    def forward(
        self,
        wcode,
        noise=None,
    ):
        '''Too complex forward, it is stupid'''
        if noise is None:
            noise = [None] * self.num_layers

        # wcode = self.style(zcode)
        # self.n_latent -- 18
        if wcode.ndim < 3:
            latent = wcode.unsqueeze(1).repeat(1, self.n_latent, 1)
        else:
            latent = wcode

        # (Pdb) latent.size()
        # torch.Size([1, 18, 512])
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        # [::2] -- start 0, step 2, --> 0, 2, 4, 6, 8 ...
        # [1::2] -- start 1, step 2, --> 1, 3, 5, 7, 9 ...
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        self.last_latent = latent

        # image = skip
        '''Post image, from [-1.0, 1.0] to [0.0, 1.0].'''
        image = ((skip + 1.0)/2.0).clamp(0.0, 1.0)

        return image

def model_device():
    """Please call this function after model_setenv. """
    return torch.device(os.environ["DEVICE"])

def model_setenv():
    """Setup environ  ..."""

    # random init ...
    # random.seed(42)
    # torch.manual_seed(42)
    # Set default device to avoid exceptions
    if os.environ.get("DEVICE") != "YES" and os.environ.get("DEVICE") != "NO":
        os.environ["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.environ["DEVICE"] == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    print("Running Environment:")
    print("----------------------------------------------")
    print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVICE"])

def noise_loss(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss

def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


class StyleCodec:
    """Style VAE."""

    def __init__(self, project="stylegan2-ffhq-config-f.pth"):
        """Init."""
        self.project = project

        print("Start creating StyleCodec for {} ...".format(project))
        model_setenv()
        self.device = model_device()

        # Following meet project config ...
        self.resolution = 1024
        self.z_space_dim = 512
        self.generator = Generator(self.resolution, self.z_space_dim, 8)

        # Load checkpoint, do factorizing ...
        self.load()
        self.mean_wcode = self.generator.mean_latent(4096)

    def zcode(self, n):
        '''Generate zcode.'''
        return torch.randn(n, self.z_space_dim, device=self.device)

    def grid_image(self, tensor, nrow=2):
        '''Convert tensor to PIL Image.'''
        grid = utils.make_grid(tensor, nrow=nrow)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
            1, 2, 0).to('cpu', torch.uint8).numpy()
        image = Image.fromarray(ndarr)
        return image        

    def decode(self, wcode):
        """input: wcode format: Bxz_space_dim [-1.0, 1.0]  tensor.
           output: BxCxHxW [0, 1.0] tensor on CPU.
        """
        assert wcode.dim() == 2, "wcode must be BxS tensor."
        with torch.no_grad():
            img = self.generator(wcode)
        return img.cpu()

    def encode(self, image):
        '''input:  image with BxCxHxW [0, 1,0] Tensor
           output: Bxz_space wcode.
        '''
        return self.train(image)

    def edit(self, wcode, k, d = 5.0):
        '''Semantic edit.
            k -- semantic number
            d -- semantic offset
        '''
        return wcode + d * self.eigen(k).unsqueeze(0)

    def sample(self, number, seed=-1):
        '''Sample.'''
        if seed < 0:
            random.seed()
            random_seed = random.randint(0, 1000000)
        else:
            random_seed = seed
        torch.manual_seed(random_seed)
        
        image = self.decode(self.zcode(number))
        nrow = int(math.sqrt(number) + 0.5) 
        image = self.grid_image(image, nrow=nrow)
        return image, random_seed

    def eigen(self, index):
        # eigen vectors ...
        assert index < self.z_space_dim
        return self.eigvec[:, index]

    def load(self):
        '''Load ...'''
        print("Loading project {} ...".format(self.project))
        # workspace = os.path.dirname(inspect.getfile(self.__init__))
        # checkpoint = Path(workspace + "/models/" + self.project)
        checkpoint = Path("models/" + self.project)

        model_weights = torch.load(checkpoint)['g_ema']
        self.generator.load_state_dict(model_weights, strict=False)
        self.generator.eval()
        self.generator = self.generator.to(self.device)

        # Start weight factorizing
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
        self.eigvec = torch.svd(W).V.to(self.device)

    def __repr__(self):
        """
        Return printable string of the model.
        """
        fmt_str = '----------------------------------------------\n'
        fmt_str += 'Project: '.format(self.project) + '\n'
        fmt_str += '    Image resolution: {}\n'.format(self.resolution)
        fmt_str += '    Z space dimension: {}\n'.format(self.z_space_dim)
        fmt_str += '----------------------------------------------\n'

        return fmt_str


    def train(self, ref_images, epochs=100, start_lr = 0.1):
        '''Train ...'''

        if ref_images.dim() < 4:
            ref_images = ref_images.unsqueeze(0)
        ref_images = ref_images.to(os.environ["DEVICE"])

        ref_batch, channel, ref_height, ref_width = ref_images.shape
        assert ref_height == ref_width

        noise_var_list = []
        for noise in self.generator.make_noise():
            normal_noise = noise.repeat(ref_batch, 1, 1, 1).normal_()
            normal_noise.requires_grad = True
            noise_var_list.append(normal_noise)

        n_mean_latent = 10000
        latent_out = torch.randn(n_mean_latent, 512, device=self.device)
        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
        del noise_sample, latent_out

        latent_var = latent_mean.detach().clone().unsqueeze(0).repeat(ref_batch, 1)
        latent_var.requires_grad = True

        optimizer = optim.Adam([latent_var] + noise_var_list, lr=start_lr)

        percept = lpips.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=os.environ["DEVICE"].startswith("cuda")
        )

        progress_bar = tqdm(range(epochs))
        noise_level = 0.05
        noise_ramp = 0.07
        self.generator.train()

        for i in progress_bar:
            t = i / epochs

            lr = get_lr(t, start_lr)
            optimizer.param_groups[0]["lr"] = lr

            # Inject Noise to latent_n
            noise_strength = latent_std * noise_level * max(0, 1 - t / noise_ramp) ** 2
            latent_n = latent_var + torch.randn_like(latent_var) * noise_strength.item()
            gen_images = self.generator(latent_n, noise=noise_var_list)

            batch, channel, height, width = gen_images.shape
            if height > ref_height:
                factor = height // ref_height
                gen_images = gen_images.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                gen_images = gen_images.mean([3, 5])

            p_loss = percept(gen_images, ref_images).sum()
            n_loss = noise_loss(noise_var_list)
            mse_loss = F.mse_loss(gen_images, ref_images)

            loss = p_loss + 1e5 * n_loss + 0.2 * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noise_var_list)

            progress_bar.set_description(
                (
                    f"Loss = perceptual: {p_loss.item():.4f}; noise: {n_loss.item():.4f};"
                    f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                )
            )
        last_latent = latent_n.detach().clone()

        self.generator.eval()

        del noise_var_list
        torch.cuda.empty_cache()

        # maybe the best latent ?
        return last_latent

