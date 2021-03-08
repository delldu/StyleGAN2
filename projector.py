import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import lpips
from model import Generator

import pdb


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument(
        "--ckpt", type=str, default="models/stylegan2-ffhq-config-f.pth", help="path to the model checkpoint"
    )
    parser.add_argument(
        "--image-size", type=int, default=32, help="input image sizes of the generator"
    )

    parser.add_argument(
        "--size", type=int, default=1024, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--step", type=int, default=100, help="optimize iterations")
    parser.add_argument("--mse", type=float, default=0.01, help="weight of the mse loss")
    parser.add_argument(
        "files", metavar="FILES", nargs="+", help="path to image files to be projected"
    )

    args = parser.parse_args()

    resize = min(args.size, args.image_size)

    # Low resolution transformer
    lrtransform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
        ]
    )
    hrtransform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
        ]
    )

    imgs = []

    for imgfile in args.files:
        img = Image.open(imgfile).convert("RGB")
        lrimg = lrtransform(img)
        lrimg.save("sample/lr-{}".format(os.path.basename(imgfile)))

        img = hrtransform(img)
        imgs.append(img)

    imgs = torch.stack(imgs, 0).to(device)

    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    with torch.no_grad():
        latent_in = g_ema.mean_latent(8196)
    latent_in.requires_grad = True

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    optimizer = optim.Adam([latent_in], lr=args.lr)

    pbar = tqdm(range(args.step))
    latent_path = []

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength)

        img_gen = g_ema(latent_n, noise=None)

        batch, channel, height, width = img_gen.shape

        if height > resize:
            factor = height // resize

            img_gen = img_gen.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen = img_gen.mean([3, 5])

        p_loss = percept(img_gen, imgs).sum()
        mse_loss = F.mse_loss(img_gen, imgs)

        loss = p_loss + args.mse * mse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0 or (i + 1) == args.step:
            latent_path.append(latent_in.detach().clone())

        pbar.set_description(
            (
                f"perceptual: {p_loss.item():.4f}; "
                f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
            )
        )

    img_gen = g_ema(latent_path[-1], noise=None)

    img_ar = make_image(img_gen)

    # result_file = {}
    for i, imgfile in enumerate(args.files):
        img_name = "sample/hr-{}".format(os.path.basename(imgfile))
        pil_img = Image.fromarray(img_ar[i])
        pil_img.save(img_name)

