import hfai_env
hfai_env.set_env("weather")

import os, sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from functools import partial

import timm.optim
from timm.scheduler import create_scheduler

import hfai
hfai.set_watchdog_time(21600)
import hfai.nccl.distributed as dist
# from hfai.nn.parallel import DistributedDataParallel
from torch.nn.parallel import DistributedDataParallel
from ffrecord.torch import DataLoader
import hfai.nn as hfnn
from hfai.datasets import ERA5

from model.afnonet import AFNONet
from utils.params import get_args
from utils.tools import getModelSize, load_model, save_model
from utils.eval import single_step_evaluate

SAVE_PATH = Path('./output/fourcastnet/')
SAVE_PATH.mkdir(parents=True, exist_ok=True)


def train_one_epoch(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler, lr_scheduler, min_loss):
    model.train()
    accumulation_steps = 8

    for step, batch in enumerate(data_loader):
        if step < start_step:
            continue

        xt, xt1 = [x.half().cuda(non_blocking=True) for x in batch]
        x = xt.transpose(3, 2).transpose(2, 1)
        y = xt1.transpose(3, 2).transpose(2, 1)

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)
            loss /= accumulation_steps
        loss_scaler.scale(loss).backward()

        # 梯度累积
        if (step+1) % accumulation_steps == 0:
            loss_scaler.step(optimizer)
            loss_scaler.update()
            optimizer.zero_grad()

        if dist.get_rank() == 0 and hfai.receive_suspend_command():
            save_model(model, epoch, step+1, optimizer, lr_scheduler, loss_scaler, min_loss, SAVE_PATH/'pretrain_latest.pt')
            time.sleep(5)
            hfai.go_suspend()


def main(local_rank):
    args = get_args()
    args.epochs = 80
    args.batch_size = 4
    args.lr = 5e-4
    # input size
    h, w = 720, 1440
    x_c, y_c = 20, 20

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # init dist
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", "54247")
    hosts = int(os.environ.get("WORLD_SIZE", "1"))  # number of nodes
    rank = int(os.environ.get("RANK", "0"))  # node id
    gpus = torch.cuda.device_count()  # gpus per node

    dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts * gpus, rank=rank * gpus + local_rank)
    torch.cuda.set_device(local_rank)

    train_dataset = ERA5(split="train", mode='pretrain', check_data=True)
    train_datasampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset, args.batch_size, sampler=train_datasampler, num_workers=8, pin_memory=True, drop_last=True)

    val_dataset = ERA5(split="val", mode='pretrain', check_data=True)
    val_datasampler = DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(val_dataset, args.batch_size, sampler=val_datasampler, num_workers=8, pin_memory=True, drop_last=False)

    model = AFNONet(img_size=[h, w], in_chans=x_c, out_chans=y_c, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model = hfnn.to_hfai(model)

    if local_rank == 0:
        param_sum, buffer_sum, all_size = getModelSize(model)
        print(f"Rank: {rank}, Local_rank: {local_rank} | Number of Parameters: {param_sum}, Number of Buffers: {buffer_sum}, Size of Model: {all_size:.4f} MB\n")

    model = DistributedDataParallel(model.cuda(), device_ids=[local_rank])

    # args.lr = args.lr * args.batch_size * dist.get_world_size() / 512.0
    param_groups = timm.optim.optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = torch.cuda.amp.GradScaler(enabled=True)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = nn.MSELoss()

    # load
    start_epoch, start_step, min_loss = load_model(model.module, optimizer, lr_scheduler, loss_scaler, SAVE_PATH/'pretrain_latest.pt')
    if local_rank == 0:
        print(f"Start training for {args.epochs} epochs")

    for epoch in range(start_epoch, args.epochs):

        train_one_epoch(epoch, start_step, model, criterion, train_dataloader, optimizer, loss_scaler, lr_scheduler, min_loss)
        start_step = 0
        lr_scheduler.step(epoch)

        train_loss = single_step_evaluate(train_dataloader, model, criterion)
        val_loss = single_step_evaluate(val_dataloader, model, criterion)

        if rank == 0 and local_rank == 0:
            print(f"Epoch {epoch} | Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
            if val_loss < min_loss:
                min_loss = val_loss
                save_model(model, path=SAVE_PATH / 'backbone.pt', only_model=True)
            save_model(model, epoch+1, 0, optimizer, lr_scheduler, loss_scaler, min_loss, SAVE_PATH / 'pretrain_latest.pt')


if __name__ == '__main__':
    ngpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)

