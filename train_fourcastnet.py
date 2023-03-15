import hfai
hfai.set_watchdog_time(21600)

import os
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from functools import partial
import hfai.nccl.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import timm.optim
from timm.scheduler import create_scheduler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from data_factory.datasets import ERA5
from model.afnonet import AFNONet
from utils.params import get_args
from utils.tools import getModelSize, load_model, save_model
from utils.eval import fourcastnet_pretrain_evaluate, fourcastnet_finetune_evaluate

SAVE_PATH = Path('./output/model/fourcastnet/')
SAVE_PATH.mkdir(parents=True, exist_ok=True)


def pretrain_one_epoch(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler, lr_scheduler, min_loss):
    loss_val = torch.tensor(0., device="cuda")
    count = torch.tensor(1e-5, device="cuda")

    model.train()

    for step, batch in enumerate(data_loader):
        if step < start_step:
            continue

        _, x, y = [x.half().cuda(non_blocking=True) for x in batch]
        x = x.transpose(3, 2).transpose(2, 1)
        y = y.transpose(3, 2).transpose(2, 1)

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)
            if torch.isnan(loss).int().sum() == 0:
                count += 1
                loss_val += loss

        loss_scaler.scale(loss).backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()
        optimizer.zero_grad()

        if dist.get_rank() == 0 and hfai.client.receive_suspend_command():
            save_model(model, epoch, step+1, optimizer, lr_scheduler, loss_scaler, min_loss, SAVE_PATH/'pretrain_latest.pt')
            hfai.client.go_suspend()

    return loss_val.item() / count.item()


def finetune_one_epoch(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler, lr_scheduler, min_loss):
    loss_val = torch.tensor(0., device="cuda")
    count = torch.tensor(1e-5, device="cuda")

    model.train()

    for step, batch in enumerate(data_loader):
        if step < start_step:
            continue

        xt0, xt1, xt2 = [x.half().cuda(non_blocking=True) for x in batch]
        xt0 = xt0.transpose(3, 2).transpose(2, 1)
        xt1 = xt1.transpose(3, 2).transpose(2, 1)
        xt2 = xt2.transpose(3, 2).transpose(2, 1)

        with torch.cuda.amp.autocast():
            out = model(xt0)
            loss = criterion(out, xt1)
            out = model(out)
            loss += criterion(out, xt2)
            if torch.isnan(loss).int().sum() == 0:
                count += 1
                loss_val += loss

        loss_scaler.scale(loss).backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()
        optimizer.zero_grad()

        if dist.get_rank() == 0 and hfai.client.receive_suspend_command():
            save_model(model, epoch, step + 1, optimizer, lr_scheduler, loss_scaler, min_loss, SAVE_PATH / 'finetune_latest.pt')
            hfai.go_suspend()

    return loss_val.item() / count.item()


def train(local_rank, rank, args):
    # input size
    h, w = 720, 1440
    x_c, y_c = 24, 20

    model = AFNONet(img_size=[h, w], in_chans=x_c, out_chans=y_c, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))
    model = hfai.nn.to_hfai(model)
    if local_rank == 0:
        param_sum, buffer_sum, all_size = getModelSize(model)
        print(f"Number of Parameters: {param_sum}, Number of Buffers: {buffer_sum}, Size of Model: {all_size:.4f} MB")
    model = DistributedDataParallel(model.cuda(), device_ids=[local_rank])

    param_groups = timm.optim.optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = torch.cuda.amp.GradScaler(enabled=True)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = torch.nn.MSELoss()

    train_dataset = ERA5(split="train", check_data=True, modelname='fourcastnet')
    train_datasampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = train_dataset.loader(args.batch_size, sampler=train_datasampler, num_workers=8, pin_memory=True, drop_last=False)
    val_dataset = ERA5(split="val", check_data=True, modelname='fourcastnet')
    val_datasampler = DistributedSampler(val_dataset)
    val_dataloader = val_dataset.loader(args.batch_size, sampler=val_datasampler, num_workers=8, pin_memory=True, drop_last=False)

    # load
    start_epoch, start_step, min_loss = load_model(model.module, optimizer, lr_scheduler, loss_scaler, SAVE_PATH / 'pretrain_latest.pt')
    if local_rank == 0:
        print(f"Start pretrain for {args.pretrain_epochs} epochs")

    for epoch in range(start_epoch, args.pretrain_epochs):

        train_loss = pretrain_one_epoch(epoch, start_step, model, criterion, train_dataloader, optimizer, loss_scaler, lr_scheduler, min_loss)
        start_step = 0
        lr_scheduler.step(epoch)

        val_loss = fourcastnet_pretrain_evaluate(val_dataloader, model, criterion)

        if rank == 0 and local_rank == 0:
            print(f"Epoch {epoch} | Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
            if val_loss < min_loss:
                min_loss = val_loss
                save_model(model, path=SAVE_PATH / 'backbone.pt', only_model=True)
            save_model(model, epoch + 1, 0, optimizer, lr_scheduler, loss_scaler, min_loss, SAVE_PATH / 'pretrain_latest.pt')


    # load
    start_epoch, start_step, min_loss = load_model(model.module, optimizer, lr_scheduler, loss_scaler, SAVE_PATH / 'finetune_latest.pt')
    if local_rank == 0:
        print(f"Start finetune for {args.finetune_epochs} epochs")

    for epoch in range(start_epoch, args.finetune_epochs):

        train_loss = finetune_one_epoch(epoch, start_step, model, criterion, train_dataloader, optimizer, loss_scaler, lr_scheduler, min_loss)
        start_step = 0
        lr_scheduler.step(epoch)

        val_loss = fourcastnet_finetune_evaluate(val_dataloader, model, criterion)

        if rank == 0 and local_rank == 0:
            print(f"Epoch {epoch} | Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
            if val_loss < min_loss:
                min_loss = val_loss
                save_model(model, path=SAVE_PATH / 'backbone.pt', only_model=True)
            save_model(model, epoch + 1, 0, optimizer, lr_scheduler, loss_scaler, min_loss, SAVE_PATH / 'finetune_latest.pt')


def main(local_rank, args):
    # fix the seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    cudnn.benchmark = True

    # init dist
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", "54247")
    hosts = int(os.environ.get("WORLD_SIZE", "1"))  # number of nodes
    rank = int(os.environ.get("RANK", "0"))  # node id
    gpus = torch.cuda.device_count()  # gpus per node

    dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts * gpus, rank=rank * gpus + local_rank)
    torch.cuda.set_device(local_rank)

    train(local_rank, rank, args)


if __name__ == '__main__':
    args = get_args()
    args.epochs = 200
    args.batch_size = 2
    args.lr = 5e-4

    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(args,), nprocs=ngpus, bind_numa=True)



