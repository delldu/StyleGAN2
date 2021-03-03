"""Create model."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 03月 02日 星期二 12:48:05 CST
# ***
# ************************************************************************************/
#

import torch
import torch.nn as nn


class GanEncoderModel(nn.Module):
    """GanEncoder Model."""

    def __init__(self):
        """Init model."""

        super(GanEncoderModel, self).__init__()

    def forward(self, x):
        """Forward."""

        return x


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

    def zcode(self, n):
        '''Generate zcode.'''
        return torch.randn(n, self.z_space_dim, device=self.device)


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

