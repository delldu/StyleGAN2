"""Data loader."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 03月 02日 星期二 12:48:05 CST
# ***
# ************************************************************************************/
#

import os
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.utils as utils
from tqdm import tqdm
import random
import lmdb
from io import BytesIO
import numpy as np
from stylegan2 import get_decoder
from model import model_device, model_setenv

import pdb

dataset_dirname = "dataset"
train_dataset_file = "{}/train".format(dataset_dirname)
test_dataset_file = "{}/test".format(dataset_dirname)
image_size = 224

def image_to_bytes(image):
    '''Transform PIL image to bytes'''

    buffer = BytesIO()
    image.save(buffer, format="jpeg", quality=100)
    return buffer.getvalue()

def bytes_to_image(bytes):
    '''Transform bytes to PIL image'''

    buffer = BytesIO(bytes)
    return Image.open(buffer).convert("RGB")

def label_to_bytes(label):
    '''Transform label = torch.randn(1, 512) to bytes'''

    buffer = BytesIO()
    np.save(buffer, label.numpy())
    return buffer.getvalue()

def bytes_to_label(bytes):
    ''' Save label to lmdb, label = torch.randn(1, 512)'''

    buffer = BytesIO(bytes)
    label = np.load(buffer)

    return torch.from_numpy(label)


def sample_label(seed=-1):
    '''Sample.'''
    if seed < 0:
        random.seed()
        random_seed = random.randint(0, 1000000)
    else:
        random_seed = seed
    torch.manual_seed(random_seed)

    return torch.randn(1, 512)

def create_database(dbname, total):
    ''' Create database.'''
    model_setenv()


    model = get_decoder()
    device = model_device()
    model = model.to(device)
    model.eval()

    toimage = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(image_size),
        ]
    )
    print("Creating dataset {} ...".format(dbname))

    progress_bar = tqdm(total = total)
    with lmdb.open(dbname, map_size=1024 ** 4, readahead=False) as env:
        # prepare(env, imgset, args.n_worker, sizes=sizes, resample=resample)
        for i in range(total):
            progress_bar.update(1)

            label_key = "l-{:09d}".format(i)
            image_key = "i-{:09d}".format(i)

            label_tensor = sample_label()
            with torch.no_grad():
                label_tensor = model.style(label_tensor.to(device))
                image_tensor = model(label_tensor)
            image_tensor = image_tensor.cpu().squeeze()
            label_tensor = label_tensor.cpu()

            image = toimage(image_tensor)

            # image.show()
            image = image_to_bytes(image)
            label = label_to_bytes(label_tensor)

            with env.begin(write=True) as txn:
                txn.put(image_key.encode("utf-8"), image)
                txn.put(label_key.encode("utf-8"), label)

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))

    del model
    print("Create database {} OK.".format(dbname))


def grid_image(tensor_list, nrow=3):
    grid = utils.make_grid(
        torch.cat(tensor_list, dim=0), nrow=nrow)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
        1, 2, 0).to('cpu', torch.uint8).numpy()
    image = Image.fromarray(ndarr)
    return image

def get_transform(train=True):
    """Transform images."""
    ts = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    return ts


class GanEncoderDataset(data.Dataset):
    """Define dataset."""

    def __init__(self, dbname, transforms=get_transform()):
        """Init dataset."""
        super(GanEncoderDataset, self).__init__()

        self.dbname = dbname
        self.transforms = transforms

        self.env = lmdb.open(dbname, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            raise IOError('Cannot open lmdb dataset', dbname)

        with self.env.begin(write=False) as txn:
            self.total_numbers = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __getitem__(self, index):
        """Load images."""

        with self.env.begin(write=False) as txn:
            label_key = "l-{:09d}".format(index)
            image_key = "i-{:09d}".format(index)
            label_bytes = txn.get(label_key.encode('utf-8'))
            image_bytes = txn.get(image_key.encode('utf-8'))

        label = bytes_to_label(label_bytes)
        image = bytes_to_image(image_bytes)
        image = self.transforms(image)

        return image, label

    def __len__(self):
        """Return total numbers."""
        return self.total_numbers

    def __repr__(self):
        """
        Return printable representation of the dataset object.
        """

        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {}\n'.format(self.__len__())
        fmt_str += '    Dataset Nmae: {}\n'.format(self.dbname)
        tmp = '    Transforms: '
        fmt_str += '{0}{1}\n'.format(tmp, self.transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def train_data(bs):
    """Get data loader for trainning & validating, bs means batch_size."""

    train_ds = GanEncoderDataset(train_dataset_file, get_transform(train=True))
    print(train_ds)

    # Split train_ds in train and valid set
    valid_len = int(0.2 * len(train_ds))
    indices = [i for i in range(len(train_ds) - valid_len, len(train_ds))]

    valid_ds = data.Subset(train_ds, indices)
    indices = [i for i in range(len(train_ds) - valid_len)]
    train_ds = data.Subset(train_ds, indices)

    # Define training and validation data loaders
    train_dl = data.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4)
    valid_dl = data.DataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=4)

    return train_dl, valid_dl

def test_data(bs):
    """Get data loader for test, bs means batch_size."""

    test_ds = GanEncoderDataset(test_dataset_file, get_transform(train=False))
    test_dl = data.DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4)

    return test_dl


def get_data(trainning=True, bs=4):
    """Get data loader for trainning & validating, bs means batch_size."""

    return train_data(bs) if trainning else test_data(bs)

def GanEncoderDatasetTest():
    """Test dataset ..."""

    ds = GanEncoderDataset(test_dataset_file, get_transform(train=False))
    print(ds)

    # src, tgt = ds[0]
    # grid = utils.make_grid(torch.cat([src.unsqueeze(0), tgt.unsqueeze(0)], dim=0), nrow=2)
    # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # image = Image.fromarray(ndarr)
    # image.show()

if __name__ == '__main__':
    """Create train/test database ..."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--create', help="Create dataset", action='store_true')
    parser.add_argument('--test', help="Test dataset", action='store_true')

    args = parser.parse_args()

    if not os.path.exists(dataset_dirname):
        os.makedirs(dataset_dirname)

    if args.create:
        create_database(train_dataset_file, 4096)
        create_database(test_dataset_file, 256)

    if args.test:
        GanEncoderDatasetTest()
