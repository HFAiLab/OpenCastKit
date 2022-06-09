import torch
import hfai.nccl.distributed as dist


@torch.no_grad()
def single_step_evaluate(data_loader, model, criterion):
    loss, total = torch.zeros(2).cuda()

    # switch to evaluation mode
    model.eval()
    for step, batch in enumerate(data_loader):
        x, y = [x.float().cuda(non_blocking=True) for x in batch[:2]]
        x = x.transpose(3, 2).transpose(2, 1)
        y = y.transpose(3, 2).transpose(2, 1)

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
        x, y_0, y_1 = [x.float().cuda(non_blocking=True) for x in batch[:3]]
        x = x.transpose(3, 2).transpose(2, 1)
        y_0 = y_0.transpose(3, 2).transpose(2, 1)
        y_1 = y_1.transpose(3, 2).transpose(2, 1)

        out_0 = model(x)
        loss_0 = criterion(out_0, y_0)
        out_1 = model(out_0)
        loss_1 = criterion(out_1, y_1)

        loss += torch.add(loss_0, loss_1)

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
        x, _, _, y = [x.cuda(non_blocking=True) for x in batch[[0, 3]]]
        x = x.transpose(3, 2).transpose(2, 1)
        y = torch.unsqueeze(y, 1)

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