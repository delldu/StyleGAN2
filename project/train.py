"""Model trainning & validating."""
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
import torch.optim as optim
from data import get_data
from model import model_setenv, model_device, model_save, train_epoch, valid_epoch
from encoder import get_encoder

if __name__ == "__main__":
    """Trainning model."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--outputdir', type=str,
                        default="output", help="output directory")
    # parser.add_argument('--checkpoint', type=str,
    #                     default="models/ImageGanEncoder.pth", help="checkpoint file")
    parser.add_argument('--bs', type=int, default=32, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()

    # Create directory to store weights
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    # get model
    model_setenv()
    model = get_encoder()
    device = model_device()
    model = model.to(device)

    # construct optimizer and learning rate scheduler,
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr,
                          momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.1)

    # get data loader
    train_dl, valid_dl = get_data(trainning=True, bs=args.bs)

    for epoch in range(args.epochs):
        if epoch % 100 == 0:
            print("Epoch {}/{}, learning rate: {} ...".format(epoch + 1, args.epochs, lr_scheduler.get_last_lr()))

        train_epoch(train_dl, model, optimizer, device, tag='train')

        valid_epoch(valid_dl, model, device, tag='valid')

        lr_scheduler.step()

        if epoch == (args.epochs // 2) or (epoch == args.epochs - 1):
            model_save(model, os.path.join(args.outputdir, "latest-checkpoint-{:04d}.pth".format(epoch)))
