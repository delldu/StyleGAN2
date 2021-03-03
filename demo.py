import argparse
import os

import torch
from torchvision import transforms as T
from PIL import Image
from model import StyleCodec

import pdb

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

    # Test samples ...
    images, seed = model.sample(9)
    images.save("sample/sample-9.png")

    # Test projector and edit


    # for imgfile in args.files:
    #     print("Projecting and edit {} ...".format(imgfile))

    #     img = Image.open(imgfile).convert("RGB")
    #     lrimg = normal_size_transform(img)
    #     lrimg.save("sample/lr-{}".format(os.path.basename(imgfile)))

    #     img = normal_tensor_transform(lrimg).unsqueeze(0)
    #     wcode = model.encode(img)

    #     imgs = []
    #     for j in range(-1, 2):
    #         ncode = model.edit(wcode, args.index, -5.0*j)
    #         img_gen = model.decode(ncode)
    #         imgs.append(img_gen)

    #     imgs = torch.cat(imgs, dim=0)
    #     image = model.grid_image(imgs, nrow = 3)
    #     filename = "sample/" + os.path.basename(imgfile) + "-projector.png"
    #     image.save(filename)


    # zcode = model.zcode(3)
    # wcode = model.to_wcode(zcode)
    # wcode1 = model.edit(wcode, args.index, -5.0)
    # wcode2 = model.edit(wcode, args.index, +5.0)

    # img1 = model.decode(wcode1)
    # img = model.decode(wcode)
    # img2 = model.decode(wcode2)
    # imgs = torch.cat([img1, img, img2], dim=0)
    # image = model.grid_image(imgs, nrow = 3)
    # image.save("sample/edit.png")

