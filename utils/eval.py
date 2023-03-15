import torch
import hfai.nccl.distributed as dist


@torch.no_grad()
def fourcastnet_pretrain_evaluate(data_loader, model, criterion):
    loss = torch.tensor(0., device="cuda")
    count = torch.tensor(1e-5, device="cuda")

    # switch to evaluation mode
    model.eval()
    for batch in data_loader:
        _, x, y = [x.half().cuda(non_blocking=True) for x in batch]
        x = x.transpose(3, 2).transpose(2, 1)
        y = y.transpose(3, 2).transpose(2, 1)

        with torch.cuda.amp.autocast():
            out = model(x)
            tmp_loss = criterion(out, y)
            if torch.isnan(tmp_loss).int().sum() == 0:
                count += 1
                loss += tmp_loss

    dist.reduce(loss, 0)
    dist.reduce(count, 0)

    loss_val = 0
    if dist.get_rank() == 0:
        loss_val = loss.item() / count.item()
    return loss_val


@torch.no_grad()
def fourcastnet_finetune_evaluate(data_loader, model, criterion):
    loss = torch.tensor(0., device="cuda")
    count = torch.tensor(1e-5, device="cuda")

    # switch to evaluation mode
    model.eval()
    for batch in data_loader:
        xt0, xt1, xt2 = [x.half().cuda(non_blocking=True) for x in batch]
        xt0 = xt0.transpose(3, 2).transpose(2, 1)
        xt1 = xt1.transpose(3, 2).transpose(2, 1)
        xt2 = xt2.transpose(3, 2).transpose(2, 1)

        with torch.cuda.amp.autocast():
            out = model(xt0)
            loss += criterion(out, xt1)
            out = model(out)
            loss += criterion(out, xt2)
        count += 1

    dist.reduce(loss, 0)
    dist.reduce(count, 0)

    loss_val = 0
    if dist.get_rank() == 0:
        loss_val = loss.item() / count.item()
    return loss_val


@torch.no_grad()
def graphcast_evaluate(data_loader, graph, model, criterion, dp_group, pp_group):
    is_last_pipeline_stage = (pp_group.rank() == pp_group.size() - 1)
    loss = torch.tensor(0., device="cuda")
    count = torch.tensor(0., device="cuda")

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

    # switch to evaluation mode
    model.eval()
    for batch in data_loader:
        x, y = [x.half().cuda(non_blocking=True) for x in batch]
        input_x[0] = x

        with torch.cuda.amp.autocast():
            out = model(*input_x)

            if is_last_pipeline_stage:
                loss += criterion(out, y)
                count += 1

    # all-reduce in data paralel group
    if is_last_pipeline_stage:
        dist.all_reduce(loss, group=dp_group)
        dist.all_reduce(count, group=dp_group)
        loss = loss / count
    else:
        loss = torch.tensor(0., device="cuda")

    # broadcast from the last stage to other pipeline stages
    dist.all_reduce(loss, group=pp_group)

    return loss.item()
