"""Model predict."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 03月 02日 星期二 12:48:05 CST
# ***
# ************************************************************************************/
#
import argparse
import glob
import os
import pdb
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from stylegan2_encoder import get_encoder
from stylegan2_decoder import get_decoder

from model import model_setenv, model_device

if __name__ == "__main__":
    """Predict."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default="dataset/input/*.png", help="input image")
    parser.add_argument('--output', type=str, default="output", help="output directory")
    args = parser.parse_args()

    # Create directory to store results
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model_setenv()
    encoder = get_encoder()
    device = model_device()
    encoder = encoder.to(device)
    encoder.eval()

    decoder = get_decoder()
    decoder = decoder.to(device)
    decoder.eval()

    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    image_filenames = sorted(glob.glob(args.input))
    progress_bar = tqdm(total=len(image_filenames))

    for index, filename in enumerate(image_filenames):
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")
        input_tensor = totensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            wcode = encoder(input_tensor)
            output_tensor = decoder(wcode).squeeze()

        toimage(output_tensor.cpu()).save(args.output + "/output_" + os.path.basename(filename))
