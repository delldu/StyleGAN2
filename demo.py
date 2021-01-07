import argparse
import os

import torch
from torchvision import transforms as T
from PIL import Image
from model import StyleCodec

import pdb

def noise_regularize(noises):
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


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


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
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument(
        "--image-size", type=int, default=32, help="input image sizes of the generator"
    )

    parser.add_argument(
        "--size", type=int, default=1024, help="output image sizes of the generator"
    )
    parser.add_argument(
        "files", metavar="FILES", nargs="+", help="path to image files to be projected"
    )

    parser.add_argument(
        "--index", type=int, default=1, help="Edit index -- Semantic number"
    )    

    args = parser.parse_args()

    resize = min(args.size, args.image_size)
    # Size transformer
    normal_size_transform = T.Compose(
        [
            T.Resize(resize),
            T.CenterCrop(resize),
        ]
    )
    # Tensor transform
    normal_tensor_transform = T.Compose(
        [
            T.ToTensor(), 
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )


    model = StyleCodec()

    for imgfile in args.files:
        print("Projecting {} ...".format(imgfile))

        img = Image.open(imgfile).convert("RGB")
        lrimg = normal_size_transform(img)
        lrimg.save("sample/lr-{}".format(os.path.basename(imgfile)))

        img = normal_tensor_transform(lrimg).unsqueeze(0)
        wcode = model.encode(img)
        img_gen = model.decode(wcode)
        filename = "sample/" + os.path.basename(imgfile) + "-projector.png"
        image = T.ToPILImage()(img_gen[0])
        image.save(filename)

    # Test samples ...
    images, seed = model.sample(9)
    images.save("sample/sample-9.png")

    zcode = model.zcode(3)
    wcode = model.to_wcode(zcode)
    wcode1 = model.edit(wcode, args.index, -5.0)
    wcode2 = model.edit(wcode, args.index, +5.0)

    img1 = model.decode(wcode1, truncation=0.7)
    img = model.decode(wcode, truncation=0.7)
    img2 = model.decode(wcode2, truncation=0.7)
    imgs = torch.cat([img1, img, img2], dim=0)
    image = model.grid_image(imgs, nrow = 3)
    image.save("sample/edit.png")

