"""Model test."""
# coding=utf-8
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
import os

import torch
from data import get_data
from model import model_setenv, valid_epoch, model_device
from encoder import get_encoder

if __name__ == "__main__":
    """Test model."""

    parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint', type=str,
    #                     default="models/ImageGanEncoder.pth", help="checkpoint file")
    parser.add_argument('--bs', type=int, default=2, help="batch size")
    args = parser.parse_args()

    # get model
    model_setenv()
    model = get_encoder()
    device = model_device()
    model = model.to(device)

    print("Start testing ...")
    test_dl = get_data(trainning=False, bs=args.bs)
    valid_epoch(test_dl, model, device, tag='test')
