import numpy as np
import torch


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()

    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()

    all_size = (param_size + buffer_size) / 1024 / 1024
    return param_sum, buffer_sum, all_size


def load_model(model, optimizer=None, lr_scheduler=None, loss_scaler=None, path=None, only_model=False):

    start_epoch, start_step = 0, 0
    min_loss = np.inf
    if path.exists():
        ckpt = torch.load(path, map_location="cpu")

        if only_model:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            loss_scaler.load_state_dict(ckpt['loss_scaler'])
            start_epoch = ckpt["epoch"]
            start_step = ckpt["step"]
            min_loss = ckpt["min_loss"]

    return start_epoch, start_step, min_loss


def save_model(model, epoch=0, step=0, optimizer=None, lr_scheduler=None, loss_scaler=None, min_loss=0, path=None, only_model=False):

    if only_model:
        states = {
            'model': model.module.state_dict(),
        }
    else:
        states = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'loss_scaler': loss_scaler.state_dict(),
            'epoch': epoch,
            'step': step,
            'min_loss': min_loss
        }

    torch.save(states, path)

