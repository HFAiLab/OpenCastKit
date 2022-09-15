import torch
import hfai.nccl.distributed as dist


@torch.no_grad()
def single_step_evaluate(data_loader, model, criterion):
    loss, total = torch.zeros(2).cuda()

    # switch to evaluation mode
    model.eval()
    for step, batch in enumerate(data_loader):
        xt, xt1 = [x.half().cuda(non_blocking=True) for x in batch]
        x = xt.transpose(3, 2).transpose(2, 1)
        y = xt1.transpose(3, 2).transpose(2, 1)

        with torch.cuda.amp.autocast():
            out = model(x)
            loss += criterion(out, y)

        total += 1

    for x in [loss, total]:
        dist.reduce(x, 0)

    loss_val = 0
    if dist.get_rank() == 0:
        loss_val = loss.item() / total.item()
    return loss_val


@torch.no_grad()
def multi_step_evaluate(data_loader, model, criterion):
    loss, total = torch.zeros(2).cuda()

    # switch to evaluation mode
    model.eval()
    for step, batch in enumerate(data_loader):
        xt, xt1, xt2 = [x.half().cuda(non_blocking=True) for x in batch]
        xt = xt.transpose(3, 2).transpose(2, 1)
        xt1 = xt1.transpose(3, 2).transpose(2, 1)
        xt2 = xt2.transpose(3, 2).transpose(2, 1)

        with torch.cuda.amp.autocast():
            out = model(xt)
            loss = criterion(out, xt1)
            out = model(out)
            loss += criterion(out, xt2)

        total += 1

    for x in [loss, total]:
        dist.reduce(x, 0)

    loss_val = 0
    if dist.get_rank() == 0:
        loss_val = loss.item() / total.item()
    return loss_val


@torch.no_grad()
def precip_step_evaluate(data_loader, backbone_model, precip_model, criterion):
    loss, total = torch.zeros(2).cuda()

    # switch to evaluation mode
    backbone_model.eval()
    precip_model.eval()

    for step, batch in enumerate(data_loader):
        xt, pt1 = [x.half().cuda(non_blocking=True) for x in batch]
        x = xt.transpose(3, 2).transpose(2, 1)
        y = torch.unsqueeze(pt1, 1)

        with torch.cuda.amp.autocast():
            out = backbone_model(x)
            out = precip_model(out)
            loss += criterion(out, y)

        total += 1

    for x in [loss, total]:
        dist.reduce(x, 0)

    loss_val = 0
    if dist.get_rank() == 0:
        loss_val = loss.item() / total.item()
    return loss_val