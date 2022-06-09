import hfai_env
hfai_env.set_env("weather")

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from functools import partial
# torch.set_default_tensor_type(torch.HalfTensor)

import timm.optim
from timm.scheduler import create_scheduler

from model.afnonet import AFNONet
from utils.tools import getModelSize
import hfai.nn as hfnn
from utils.params import get_args

from ffrecord.torch import DataLoader
from hfai.datasets import ERA5


def test_model_build():
    args = get_args()
    args.batch_size = 3
    args.lr = 5e-4
    # get size
    h, w = 720, 1440
    x_c, y_c = 20, 20
    gpu_idx = 1

    model = AFNONet(img_size=[h, w], in_chans=x_c, out_chans=y_c, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    param_sum, buffer_sum, all_size = getModelSize(model)
    print(f"Number of Parameters: {param_sum}\nNumber of Buffers: {buffer_sum}\nSize of Model: {all_size:.4f} MB\n")
    model = hfnn.to_hfai(model)
    model.cuda(device=gpu_idx)

    param_groups = timm.optim.optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # optimizer = torch.optim.SGD(param_groups, lr=args.lr)
    loss_scaler = GradScaler(enabled=True)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = nn.MSELoss()

    x = torch.rand([1, 20, 720, 1440]).cuda(device=gpu_idx)
    y_0 = torch.rand([1, 20, 720, 1440]).cuda(device=gpu_idx)
    y_1 = torch.rand([1, 20, 720, 1440]).cuda(device=gpu_idx)

    optimizer.zero_grad()

    out = model(x)
    loss = criterion(out, y_0)
    out = model(out)
    loss += criterion(out, y_1)

    loss_scaler.scale(loss).backward()
    loss_scaler.step(optimizer)
    loss_scaler.update()
    print(loss)


def test_dataloader_from_hf():
    ds = ERA5("train", True)
    x, y_0, y_1 = ds[[0]][0]
    print(f"x: {x.shape}, y_0: {y_0.shape}, y_1: {y_1.shape}")

    scaler = ds.get_scaler()
    mean = scaler.mean
    std = scaler.std
    print(f"mean: {mean}\nstd: {std}")

    loader = DataLoader(ds, batch_size=8, num_workers=8, shuffle=True)
    batch_x, batch_y_0, batch_y_1 = next(iter(loader))
    print(f"x: {batch_x.shape}, y_0: {batch_y_0.shape}, y_1: {batch_y_1.shape}")

    # batch_x = ds.get_scaler().inverse_transform(batch_x)
    # batch_y_0 = ds.get_scaler().inverse_transform(batch_y_0)

    # criterion = torch.nn.MSELoss()
    #
    # length = len(loader)
    # bar_length = 20
    # loss, num = 0.0, 1.0
    # for batch_x, batch_y_0, batch_y_1 in loader:
    #     # 进度条
    #     hashes = '#' * int(num / length * bar_length)
    #     spaces = ' ' * (bar_length - len(hashes))
    #     sys.stdout.write("\rPercent: [%s] %d%%" % (hashes + spaces, int(num * 100 / length)))
    #     sys.stdout.flush()
    #     num += 1
    #
    #     batch_x = batch_x.transpose(3, 2).transpose(2, 1)
    #     batch_y_0 = batch_y_0.transpose(3, 2).transpose(2, 1)
    #
    #     tmp = criterion(batch_x, batch_y_0)
    #     loss += tmp.item()
    #
    # print("\n", loss / (num - 1))


if __name__ == '__main__':
    test_model_build()