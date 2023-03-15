import hfai
hfai.set_watchdog_time(21600)

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import hfai.nccl.distributed as dist
from haiscale.ddp import DistributedDataParallel
from haiscale.pipeline import PipeDream, make_subgroups, partition
from torch.utils.data.distributed import DistributedSampler
import timm.optim
from timm.scheduler import create_scheduler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from data_factory.datasets import ERA5, EarthGraph
from model.graphcast_sequential import get_graphcast_module
from utils.params import get_graphcast_args
from utils.tools import load_model, save_model
from utils.eval import graphcast_evaluate

SAVE_PATH = Path('./output/graphcast/')
SAVE_PATH.mkdir(parents=True, exist_ok=True)


def train_one_epoch(epoch, model, criterion, data_loader, graph, optimizer, lr_scheduler, min_loss, dp_group, pp_group):
    is_last_pipeline_stage = (pp_group.rank() == pp_group.size() - 1)
    loss = torch.tensor(0., device="cuda")
    count = torch.tensor(0., device="cuda")
    model.train()

    input_x = [
        None,
        graph.mesh_data.x.half().cuda(non_blocking=True),
        graph.mesh_data.edge_index.cuda(non_blocking=True),
        graph.mesh_data.edge_attr.half().cuda(non_blocking=True),
        graph.grid2mesh_data.edge_index.cuda(non_blocking=True),
        graph.grid2mesh_data.edge_attr.half().cuda(non_blocking=True),
        graph.mesh2grid_data.edge_index.cuda(non_blocking=True),
        graph.mesh2grid_data.edge_attr.half().cuda(non_blocking=True)
    ]

    for step, batch in enumerate(data_loader):

        x, y = [x.half().cuda(non_blocking=True) for x in batch]
        input_x[0] = x

        with torch.cuda.amp.autocast():
            # out = model(input_x)
            step_loss, _ = model.forward_backward(*input_x, criterion=criterion, labels=(y,))

        optimizer.step()
        optimizer.zero_grad()

        if is_last_pipeline_stage:
            loss += step_loss.sum().item()
            count += 1

        if dp_group.rank() == 0 and is_last_pipeline_stage and hfai.client.receive_suspend_command():
            save_model(model.module.module, epoch, step + 1, optimizer, lr_scheduler, min_loss, SAVE_PATH / 'latest.pt')
            hfai.go_suspend()

    # all-reduce in data paralel group
    if dp_group.rank == 0 and is_last_pipeline_stage:
        dist.all_reduce(loss, group=dp_group)
        dist.all_reduce(count, group=dp_group)
        loss = loss / count

    # broadcast from the last stage to other pipeline stages
    dist.all_reduce(loss, group=pp_group)

    return loss.item()


def train(local_rank, args):
    rank, world_size = dist.get_rank(), dist.get_world_size()

    # data parallel + pipeline parallel
    dp_group, pp_group = make_subgroups(pp_size=args.pp_size)
    dp_rank, dp_size = dp_group.rank(), dp_group.size()
    pp_rank, pp_size = pp_group.rank(), pp_group.size()
    is_last_pipeline_stage = (pp_group.rank() == pp_group.size() - 1)
    print(f"RANK {rank}: data parallel {dp_rank}/{dp_size}, pipeline parallel {pp_rank}/{pp_size}", flush=True)

    # model & criterion & optimizer
    model = get_graphcast_module(args)
    # model = hfai.nn.to_hfai(model)
    balance = [1, 1, 1, 1, 1, 1, 1, 1]
    model = partition(model, pp_group.rank(), pp_group.size(), balance=balance)

    model = DistributedDataParallel(model.cuda(), process_group=dp_group)
    model = PipeDream(model, args.chunks, process_group=pp_group)

    # args.lr = args.lr * args.batch_size * dist.get_world_size() / 512.0
    param_groups = timm.optim.optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = nn.MSELoss()

    # generate graph
    graph = EarthGraph()
    graph.generate_graph()

    # load grid data
    train_dataset = ERA5(split="train", check_data=True, modelname='graphcast')
    train_datasampler = DistributedSampler(train_dataset, num_replicas=dp_size, rank=dp_rank, shuffle=True)
    train_dataloader = train_dataset.loader(args.batch_size, sampler=train_datasampler, num_workers=8, pin_memory=True, drop_last=True)
    val_dataset = ERA5(split="val", check_data=True, modelname='graphcast')
    val_datasampler = DistributedSampler(val_dataset, num_replicas=dp_size, rank=dp_rank, shuffle=True)
    val_dataloader = val_dataset.loader(args.batch_size, sampler=val_datasampler, num_workers=8, pin_memory=True, drop_last=False)

    # load
    start_epoch, min_loss = load_model(model.module.module, optimizer, lr_scheduler, SAVE_PATH / 'latest.pt')
    if local_rank == 0:
        print(f"Start training for {args.epochs} epochs")

    for epoch in range(start_epoch, args.epochs):

        train_loss = train_one_epoch(epoch, model, criterion, train_dataloader, graph, optimizer, lr_scheduler, min_loss, dp_group, pp_group)
        lr_scheduler.step(epoch)

        val_loss = graphcast_evaluate(val_dataloader, graph, model, criterion, dp_group, pp_group)

        if is_last_pipeline_stage:
            print(f"Epoch {epoch} | Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")

        if dp_rank == 0:
            save_model(model.module.module, epoch + 1, optimizer, lr_scheduler, min_loss, SAVE_PATH / 'latest.pt')
            if val_loss < min_loss:
                min_loss = val_loss
                save_model(model.module.module, path=SAVE_PATH / 'best.pt', only_model=True)

    # synchronize all processes
    model.module.reducer.stop()
    dist.barrier()


def main(local_rank, args):
    # fix the seed for reproducibility
    torch.manual_seed(2023)
    np.random.seed(2023)
    cudnn.benchmark = True

    # init dist
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", "54247")
    hosts = int(os.environ.get("WORLD_SIZE", "1"))  # number of nodes
    rank = int(os.environ.get("RANK", "0"))  # node id
    gpus = torch.cuda.device_count()  # gpus per node

    dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts * gpus, rank=rank * gpus + local_rank)
    torch.cuda.set_device(local_rank)

    train(local_rank, args)


if __name__ == '__main__':
    args = get_graphcast_args()
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(args,), nprocs=ngpus, bind_numa=True)
