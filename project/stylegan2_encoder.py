#
"""Contains the implementation of encoder for StyleGAN2 inversion.

For more details, please check the paper:
https://arxiv.org/pdf/2004.00049.pdf

https://github.com/genforce/idinvert_pytorch.git
"""

import math
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleGAN2Encoder(nn.Module):
    """Defines the encoder network for StyleGAN2 inversion.

    NOTE: The encoder takes images with `RGB` color channels and range [-1, 1]
    as inputs, and encode the input images to W+ space of StyleGAN2.
    """

    def __init__(self,
                 resolution=256,
                 w_space_dim=512,
                 image_channels=3,
                 encoder_channels_base=64,
                 encoder_channels_max=256):
        """Initializes the encoder with basic settings.

        Args:
            resolution: The resolution of the input image.
            w_space_dim: The dimension of the disentangled latent vectors, w.
                (default: 512)
            image_channels: Number of channels of the input image. (default: 3)
            encoder_channels_base: Base factor of the number of channels used in
                residual blocks of encoder. (default: 64)
            encoder_channels_max: Maximum number of channels used in residual blocks
                of encoder. (default: 1024)

        Raises:
            ValueError: If the input `resolution` is not supported.
        """
        super().__init__()

        # Initial resolution.
        self.init_res = 4
        self.resolution = resolution
        self.w_space_dim = w_space_dim
        self.image_channels = image_channels
        self.encoder_channels_base = encoder_channels_base
        self.encoder_channels_max = encoder_channels_max
        # Blocks used in encoder.
        self.num_blocks = int(math.log2(resolution))
        # self.numblocks -- 8 for 256

        # Layers used in generator.
        self.num_layers = int(
            math.log2(self.resolution // self.init_res * 2)) * 2
        # (Pdb) self.num_layers -- 14

        in_channels = self.image_channels
        out_channels = self.encoder_channels_base
        for block_idx in range(self.num_blocks):
            if block_idx == 0:
                self.add_module(
                    f'block{block_idx}',
                    FirstBlock(in_channels=in_channels,
                               out_channels=out_channels))

            elif block_idx == self.num_blocks - 1:
                in_channels = in_channels * self.init_res * self.init_res
                out_channels = self.w_space_dim * 2 * block_idx
                self.add_module(
                    f'block{block_idx}',
                    LastBlock(in_channels=in_channels,
                              out_channels=out_channels))

            else:
                self.add_module(
                    f'block{block_idx}',
                    ResBlock(in_channels=in_channels,
                             out_channels=out_channels))
            in_channels = out_channels
            out_channels = min(out_channels * 2, self.encoder_channels_max)

        self.downsample = AveragePoolingLayer()

        # layers = [PixelNorm()]

        # for i in range(n_mlp):
        #         # bias=True, bias_init=0, lr_mul=1, activation=None
        #         layers.append(
        #                 EqualLinear(z_space_dim, z_space_dim, lr_mul=lr_mlp, activation="fused_lrelu")
        #         )

        # self.style = nn.Sequential(*layers)

        # pdb.set_trace()
        # resolution = 256
        # w_space_dim = 512
        # image_channels = 3
        # encoder_channels_base = 64
        # encoder_channels_max = 1024

    def forward(self, x):
        # (Pdb) x.size() -- torch.Size([1, 3, 256, 256])

        if x.ndim != 4 or x.shape[1:] != (
                self.image_channels, self.resolution, self.resolution):
            raise ValueError(f'The input image should be with shape [batch_size, '
                             f'channel, height, width], where '
                             f'`channel` equals to {self.image_channels}, '
                             f'`height` and `width` equal to {self.resolution}!\n'
                             f'But {x.shape} is received!')

        for block_idx in range(self.num_blocks):
            if 0 < block_idx < self.num_blocks - 1:
                x = self.downsample(x)
            x = self.__getattr__(f'block{block_idx}')(x)
        # pdb.set_trace()
        # (Pdb) x.size() -- torch.Size([1, 7168])
        # ==> 1, 14, 512 ==> expand 1, 18, 512 ...
        # 18x512= 9216
        # .view(1, *self.encode_dim)

        return x


class AveragePoolingLayer(nn.Module):
    """Implements the average pooling layer.

    Basically, this layer can be used to downsample feature maps from spatial
    domain.
    """

    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        ksize = [self.scale_factor, self.scale_factor]
        strides = [self.scale_factor, self.scale_factor]
        return F.avg_pool2d(x, kernel_size=ksize, stride=strides, padding=0)


class BatchNormLayer(nn.Module):
    """Implements batch normalization layer."""

    def __init__(self, channels, gamma=False, beta=True, decay=0.9, epsilon=1e-5):
        """Initializes with basic settings.

        Args:
            channels: Number of channels of the input tensor.
            gamma: Whether the scale (weight) of the affine mapping is learnable.
            beta: Whether the center (bias) of the affine mapping is learnable.
            decay: Decay factor for moving average operations in this layer.
            epsilon: A value added to the denominator for numerical stability.
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features=channels,
                                 affine=True,
                                 track_running_stats=True,
                                 momentum=1 - decay,
                                 eps=epsilon)
        self.bn.weight.requires_grad = gamma
        self.bn.bias.requires_grad = beta

    def forward(self, x):
        return self.bn(x)


class WScaleLayer(nn.Module):
    """Implements the layer to scale weight variable and add bias.

    NOTE: The weight variable is trained in `nn.Conv2d` layer (or `nn.Linear`
    layer), and only scaled with a constant number, which is not trainable in
    this layer. However, the bias variable is trainable in this layer.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 gain=math.sqrt(2.0)):
        super().__init__()
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = gain / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        if x.ndim == 4:
            return x * self.scale + self.bias.view(1, -1, 1, 1)
        if x.ndim == 2:
            return x * self.scale + self.bias.view(1, -1)
        raise ValueError(f'The input tensor should be with shape [batch_size, '
                         f'channel, height, width], or [batch_size, channel]!\n'
                         f'But {x.shape} is received!')


class FirstBlock(nn.Module):
  """Implements the first block, which is a convolutional block."""

  def __init__(self,
               in_channels,
               out_channels,
               activation_type='lrelu'):
    super().__init__()

    self.conv = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False)
    self.bn = BatchNormLayer(channels=out_channels)
    self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # self = FirstBlock(
    #   (conv): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    #   (bn): BatchNormLayer(
    #     (bn): BatchNorm2d(64, eps=1e-05, momentum=0.09999999999999998, affine=True, track_running_stats=True)
    #   )
    #   (activate): LeakyReLU(negative_slope=0.2, inplace=True)
    # )
    # in_channels = 3
    # out_channels = 64
    # activation_type = 'lrelu'

  def forward(self, x):
    return self.activate(self.bn(self.conv(x)))


class ResBlock(nn.Module):
  """Implements the residual block.

  Usually, each residual block contains two convolutional layers, each of which
  is followed by batch normalization layer and activation layer.
  """

  def __init__(self,
               in_channels,
               out_channels,
               wscale_gain=math.sqrt(2.0),
               activation_type='lrelu'):
    """Initializes the class with block settings.

    Args:
      in_channels: Number of channels of the input tensor fed into this block.
      out_channels: Number of channels of the output tensor.
      kernel_size: Size of the convolutional kernels.
      stride: Stride parameter for convolution operation.
      padding: Padding parameter for convolution operation.
      wscale_gain: The gain factor for `wscale` layer.
      activation_type: Type of activation. Support `linear` and `lrelu`.
    """
    super().__init__()

    # Add shortcut if needed.
    if in_channels != out_channels:
      self.add_shortcut = True
      self.conv = nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False)
      self.scale = 1.0
      self.bn = BatchNormLayer(channels=out_channels)
    else:
      self.add_shortcut = False
      self.identity = nn.Identity()

    hidden_channels = min(in_channels, out_channels)

    # First convolutional block.
    self.conv1 = nn.Conv2d(in_channels=in_channels,
                           out_channels=hidden_channels,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bias=False)
    self.scale1 = wscale_gain / math.sqrt(in_channels * 3 * 3)
    # NOTE: WScaleLayer is employed to add bias.
    self.wscale1 = WScaleLayer(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=3,
                               gain=wscale_gain)
    self.bn1 = BatchNormLayer(channels=hidden_channels)

    # Second convolutional block.
    self.conv2 = nn.Conv2d(in_channels=hidden_channels,
                           out_channels=out_channels,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bias=False)
    self.scale2 = wscale_gain / math.sqrt(hidden_channels * 3 * 3)
    self.wscale2 = WScaleLayer(in_channels=hidden_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               gain=wscale_gain)
    self.bn2 = BatchNormLayer(channels=out_channels)

    self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

  def forward(self, x):
    if self.add_shortcut:
      y = self.activate(self.bn(self.conv(x) * self.scale))
    else:
      y = self.identity(x)
    x = self.activate(self.bn1(self.wscale1(self.conv1(x) / self.scale1)))
    x = self.activate(self.bn2(self.wscale2(self.conv2(x) / self.scale2)))
    return x + y


class LastBlock(nn.Module):
  """Implements the last block, which is a dense block."""

  def __init__(self,
               in_channels,
               out_channels,
               wscale_gain=1.0):
    super().__init__()

    self.fc = nn.Linear(in_features=in_channels,
                        out_features=out_channels,
                        bias=False)
    self.scale = wscale_gain / math.sqrt(in_channels)
    self.bn = BatchNormLayer(channels=out_channels)

    # pdb.set_trace()
    # self = LastBlock(
    #   (fc): Linear(in_features=16384, out_features=7168, bias=False)
    #   (bn): BatchNormLayer(
    #     (bn): BatchNorm2d(7168, eps=1e-05, momentum=0.09999999999999998, affine=True, track_running_stats=True)
    #   )
    # )
    # in_channels = 16384
    # out_channels = 7168

  def forward(self, x):
    x = x.view(x.shape[0], -1)
    x = self.fc(x) * self.scale
    x = x.view(x.shape[0], x.shape[1], 1, 1)
    return self.bn(x).view(x.shape[0], x.shape[1])

def get_encoder():
    '''Get encoder'''

    model = StyleGAN2Encoder()
    return model


def export_onnx():
    """Export onnx model."""

    import numpy as np
    import onnx
    import onnxruntime
    from onnx import optimizer

    onnx_file_name = "output/image_ganencoder.onnx"
    dummy_input = torch.randn(1, 3, 256, 256)

    # 1. Create and load model.
    torch_model = get_encoder()
    torch_model.eval()

    # 2. Model export
    print("Export model ...")

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
    # https://github.com/onnx/optimizer

    # 4. Visual model
    # python -c "import netron; netron.start('output/image_zoom.onnx')"


def verify_onnx():
    """Verify onnx model."""

    import numpy as np
    import onnxruntime

    torch_model = get_encoder()
    torch_model.eval()

    onnx_file_name = "output/image_ganencoder.onnx"
    onnxruntime_engine = onnxruntime.InferenceSession(onnx_file_name)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        torch_output = torch_model(dummy_input)
    onnxruntime_inputs = {
        onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_input)}
    onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)
    np.testing.assert_allclose(
        to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-02, atol=1e-02)
    print("Example: Onnx model has been tested with ONNXRuntime, the result looks good !")


def export_torch():
    """Export torch model."""

    script_file = "output/iamge_ganencoder.pt"

    # 1. Load model
    print("Loading model ...")
    model = get_encoder()
    model.eval()

    # 2. Model export
    print("Export model ...")
    dummy_input = torch.randn(1, 3, 256, 256)
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(script_file)


if __name__ == '__main__':
    """Onnx Tools ..."""
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--export', help="Export onnx model", action='store_true')
    parser.add_argument(
        '--verify', help="Verify onnx model", action='store_true')
    parser.add_argument('--output', type=str,
                        default="output", help="output folder")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model = get_encoder()
    print(model)

    # export_torch()

    if args.export:
        export_onnx()

    if args.verify:
        verify_onnx()
