import argparse

import torch
from torchvision import utils
from PIL import Image
from model import Generator
import pdb

def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")

    parser.add_argument(
        "-i", "--index", type=int, default=0, help="index of eigenvector"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help='channel multiplier factor. config-f = 2, else = 1',
    )
    parser.add_argument("--ckpt", type=str, 
        default="checkpoint/stylegan2-ffhq-config-f.pth",
        help="stylegan2 checkpoints")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "-n", "--n_sample", type=int, default=3, help="number of samples created"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="factor",
        help="filename prefix to result samples",
    )
    parser.add_argument(
        "--factor",
        type=str,
        default="checkpoint/factor.pth",
        help="name of the closed form factorization result factor file",
    )

    args = parser.parse_args()

    eigvec = torch.load(args.factor)["eigvec"].to(args.device)
    ckpt = torch.load(args.ckpt)
    g = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier).to(args.device)
    g.load_state_dict(ckpt["g_ema"], strict=False)

    trunc = g.mean_latent(4096)
    # (Pdb) trunc.size()
    # torch.Size([1, 512])

    zcode = torch.randn(args.n_sample, 512, device=args.device)
    latent = g.get_latent(zcode)
    # (Pdb) latent.size()
    # torch.Size([3, 512])

    for args.index in range(0, 32):
        direction = args.degree * eigvec[:, args.index].unsqueeze(0)
        # (Pdb) direction.size()
        # torch.Size([1, 512])
        img = g(
            [latent]
        )

        img1 = g(
            [latent + direction]
        )
        img2 = g(
            [latent - direction]
        )

        grid = utils.save_image(
            torch.cat([img1, img, img2], 0),
            f"sample/{args.out_prefix}_index-{args.index}_degree-{args.degree}.png",
            normalize=True,
            range=(-1, 1),
            nrow=args.n_sample,
        )

