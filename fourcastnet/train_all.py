import os
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import hfai
hfai.set_watchdog_time(21600)
import hfai.nccl.distributed as dist

import train.pretrain as pretrain
import train.fine_tune as finetune
import train.precipitation as precipitation

SAVE_PATH = Path('./output/fourcastnet/')
SAVE_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path('/weka-jd/prod/platform_team/hwj/era5/model')


def main(local_rank):

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

    pretrain.train(local_rank, rank, epoch=80, batch_size=4, lr=5e-4)
    finetune.train(local_rank, rank, epoch=25, batch_size=2, lr=1e-4)
    precipitation.train(local_rank, rank, epoch=50, batch_size=2, lr=2.5e-4)

    if dist.get_rank() == 0 and local_rank == 0:
        os.system(f'mv {SAVE_PATH}/backbone.pt {SAVE_PATH}/precipitation.pt {MODEL_PATH}/')

if __name__ == '__main__':
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=True)

