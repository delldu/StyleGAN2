import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
from PIL import Image
import pdb

def grid_image(tensor, nrow=2):
    '''Convert tensor to PIL Image.'''
    grid = utils.make_grid(tensor, nrow=nrow)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
        1, 2, 0).to('cpu', torch.uint8).numpy()
    image = Image.fromarray(ndarr)
    return image

def generate(args, g_ema, device):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)
            sample = g_ema(sample_z)
            sample = grid_image(sample, nrow = 1)
            sample.save("sample/{:06d}.png".format(i + 1))


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=10, help="number of images to be generated"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/stylegan2-ffhq-config-f.pth",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])
    g_ema.eval()

    generate(args, g_ema, device)
