import argparse

import torch
import pdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract factor/eigenvectors of latent spaces using closed form factorization"
    )

    parser.add_argument(
        "--out", type=str, default="checkpoint/factor.pth", 
        help="name of the result factor file"
    )
    parser.add_argument("--ckpt", type=str, 
        default="checkpoint/stylegan2-ffhq-config-f.pth", 
        help="name of the model checkpoint")

    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)
    # pdb.set_trace()
    # (Pdb) ckpt.keys()
    # dict_keys(['g_ema', 'latent_avg'])

    modulate = {
        k: v
        for k, v in ckpt["g_ema"].items()
        if "modulation" in k and "to_rgbs" not in k and "weight" in k
    }
    # pdb.set_trace()
    # (Pdb) modulate.keys()
    # dict_keys(['conv1.conv.modulation.weight', 
    #     'to_rgb1.conv.modulation.weight', 
    #     'convs.0.conv.modulation.weight', 
    #     'convs.1.conv.modulation.weight', 
    #     'convs.2.conv.modulation.weight', 
    #     'convs.3.conv.modulation.weight', 
    #     'convs.4.conv.modulation.weight', 
    #     'convs.5.conv.modulation.weight', 
    #     'convs.6.conv.modulation.weight', 
    #     'convs.7.conv.modulation.weight', 
    #     'convs.8.conv.modulation.weight', 
    #     'convs.9.conv.modulation.weight', 
    #     'convs.10.conv.modulation.weight', 
    #     'convs.11.conv.modulation.weight', 
    #     'convs.12.conv.modulation.weight', 
    #     'convs.13.conv.modulation.weight', 
    #     'convs.14.conv.modulation.weight', 
    #     'convs.15.conv.modulation.weight'])

    weight_mat = []
    for k, v in modulate.items():
        print(k, v.size())
        weight_mat.append(v)

    W = torch.cat(weight_mat, 0)
    eigvec = torch.svd(W).V.to("cpu")
    # pdb.set_trace()
    # len(weight_mat)--18
    # (Pdb) W.size()
    # torch.Size([6560, 512])
    # (Pdb) torch.svd(W).S.size()
    # torch.Size([512])
    # (Pdb) eigvec.size()
    # torch.Size([512, 512])

    torch.save({"ckpt": args.ckpt, "eigvec": eigvec}, args.out)

