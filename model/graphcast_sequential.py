import torch
import torch.nn as nn
from torch_scatter import scatter
from haiscale.pipeline import SequentialModel


class FeatEmbedding(torch.nn.Module):
    def __init__(self, args):
        super(FeatEmbedding, self).__init__()

        gdim, mdim, edim = args.grid_node_dim, args.mesh_node_dim, args.edge_dim
        gemb, memb, eemb = args.grid_node_embed_dim, args.mesh_node_embed_dim, args.edge_embed_dim

        # Embedding the input features
        self.grid_feat_embedding = nn.Sequential(
            nn.Linear(gdim, gemb, bias=True)
        )
        self.mesh_feat_embedding = nn.Sequential(
            nn.Linear(mdim, memb, bias=True)
        )
        self.mesh_edge_feat_embedding = nn.Sequential(
            nn.Linear(edim, eemb, bias=True)
        )
        self.grid2mesh_edge_feat_embedding = nn.Sequential(
            nn.Linear(edim, eemb, bias=True)
        )
        self.mesh2grid_edge_feat_embedding = nn.Sequential(
            nn.Linear(edim, eemb, bias=True)
        )

    def forward(self, gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x):

        bs = gx.size(0)

        gx = self.grid_feat_embedding(gx)
        mx = self.mesh_feat_embedding(mx).repeat(bs, 1, 1)
        me_x = self.mesh_edge_feat_embedding(me_x).repeat(bs, 1, 1)
        g2me_x = self.grid2mesh_edge_feat_embedding(g2me_x).repeat(bs, 1, 1)
        m2ge_x = self.mesh2grid_edge_feat_embedding(m2ge_x).repeat(bs, 1, 1)

        return gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x


class Grid2MeshEdgeUpdate(torch.nn.Module):
    def __init__(self, args):
        super(Grid2MeshEdgeUpdate, self).__init__()

        g2m_enum = args.grid2mesh_edge_num
        gemb, memb, eemb = args.grid_node_embed_dim, args.mesh_node_embed_dim, args.edge_embed_dim

        # Grid2Mesh GNN
        self.grid2mesh_edge_update = nn.Sequential(
            nn.Linear(gemb + memb + eemb, 512, bias=True),
            nn.SiLU(),
            nn.Linear(512, 64, bias=True),
            nn.SiLU(),
            nn.Linear(64, eemb, bias=True),
            nn.LayerNorm([g2m_enum, eemb])
        )

    def forward(self, gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x):

        row, col = g2me_i

        # edge update
        edge_attr_updated = torch.cat([gx[:, row], mx[:, col], g2me_x], dim=-1)
        edge_attr_updated = self.grid2mesh_edge_update(edge_attr_updated)

        # residual
        g2me_x = g2me_x + edge_attr_updated

        return gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x


class Grid2MeshNodeUpdate(torch.nn.Module):
    def __init__(self, args):
        super(Grid2MeshNodeUpdate, self).__init__()

        gnum, mnum = args.grid_node_num, args.mesh_node_num
        gemb, memb, eemb = args.grid_node_embed_dim, args.mesh_node_embed_dim, args.edge_embed_dim

        # Grid2Mesh GNN
        self.grid2mesh_node_aggregate = nn.Sequential(
            nn.Linear(memb + eemb, 512, bias=True),
            nn.SiLU(),
            nn.Linear(512, 256, bias=True),
            nn.SiLU(),
            nn.Linear(256, memb, bias=True),
            nn.LayerNorm([mnum, memb])
        )
        self.grid2mesh_grid_update = nn.Sequential(
            nn.Linear(gemb, 256, bias=True),
            nn.SiLU(),
            nn.Linear(256, gemb, bias=True),
            nn.LayerNorm([gnum, gemb])
        )

    def forward(self, gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x):

        row, col = g2me_i

        # mesh node update
        edge_agg = scatter(g2me_x, col, dim=-2, reduce='sum')
        mesh_node_updated = torch.cat([mx, edge_agg], dim=-1)
        mesh_node_updated = self.grid2mesh_node_aggregate(mesh_node_updated)

        # grid node update
        grid_node_updated = self.grid2mesh_grid_update(gx)

        # residual
        gx = gx + grid_node_updated
        mx = mx + mesh_node_updated

        return gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x


class MeshEdgeUpdate(torch.nn.Module):
    def __init__(self, args):
        super(MeshEdgeUpdate, self).__init__()

        m_enum = args.mesh_edge_num
        memb, eemb = args.mesh_node_embed_dim, args.edge_embed_dim

        # Multi-mesh GNN
        self.mesh_edge_update = nn.Sequential(
            nn.Linear(memb + memb + eemb, 512, bias=True),
            nn.SiLU(),
            nn.Linear(512, 64, bias=True),
            nn.SiLU(),
            nn.Linear(64, eemb, bias=True),
            nn.LayerNorm([m_enum, eemb])
        )

    def forward(self, gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x):

        row, col = me_i

        # edge update
        edge_attr_updated = torch.cat([mx[:, row], mx[:, col], me_x], dim=-1)
        edge_attr_updated = self.mesh_edge_update(edge_attr_updated)

        # residual
        me_x = me_x + edge_attr_updated

        return gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x


class MeshNodeUpdate(torch.nn.Module):
    def __init__(self, args):
        super(MeshNodeUpdate, self).__init__()

        mnum = args.mesh_node_num
        memb, eemb = args.mesh_node_embed_dim, args.edge_embed_dim

        # Grid2Mesh GNN
        self.mesh_node_aggregate = nn.Sequential(
            nn.Linear(memb + eemb, 512, bias=True),
            nn.SiLU(),
            nn.Linear(512, 256, bias=True),
            nn.SiLU(),
            nn.Linear(256, memb, bias=True),
            nn.LayerNorm([mnum, memb])
        )

    def forward(self, gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x):

        row, col = me_i

        # mesh node update
        edge_agg = scatter(me_x, col, dim=-2, reduce='sum')
        mesh_node_updated = torch.cat([mx, edge_agg], dim=-1)
        mesh_node_updated = self.mesh_node_aggregate(mesh_node_updated)

        # residual
        mx = mx + mesh_node_updated

        return gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x


class Mesh2GridEdgeUpdate(torch.nn.Module):
    def __init__(self, args):
        super(Mesh2GridEdgeUpdate, self).__init__()

        m2g_enum = args.mesh2grid_edge_num
        gemb, memb, eemb = args.grid_node_embed_dim, args.mesh_node_embed_dim, args.edge_embed_dim

        # Mesh2grid GNN
        self.mesh2grid_edge_update = nn.Sequential(
            nn.Linear(gemb + memb + eemb, 512, bias=True),
            nn.SiLU(),
            nn.Linear(512, 64, bias=True),
            nn.SiLU(),
            nn.Linear(64, eemb, bias=True),
            nn.LayerNorm([m2g_enum, eemb])
        )

    def forward(self, gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x):

        row, col = m2ge_i

        # edge update
        edge_attr_updated = torch.cat([mx[:, row], gx[:, col], m2ge_x], dim=-1)
        edge_attr_updated = self.mesh2grid_edge_update(edge_attr_updated)

        # residual
        m2ge_x = m2ge_x + edge_attr_updated

        return gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x


class Mesh2GridNodeUpdate(torch.nn.Module):
    def __init__(self, args):
        super(Mesh2GridNodeUpdate, self).__init__()

        gnum = args.grid_node_num
        gemb, eemb = args.grid_node_embed_dim, args.edge_embed_dim

        # Mesh2grid GNN
        self.mesh2grid_node_aggregate = nn.Sequential(
            nn.Linear(gemb + eemb, 512, bias=True),
            nn.SiLU(),
            nn.Linear(512, 256, bias=True),
            nn.SiLU(),
            nn.Linear(256, gemb, bias=True),
            nn.LayerNorm([gnum, gemb])
        )

    def forward(self, gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x):

        row, col = m2ge_i

        # mesh node update
        edge_agg = scatter(m2ge_x, col, dim=-2, reduce='sum')
        grid_node_updated = torch.cat([gx, edge_agg], dim=-1)
        grid_node_updated = self.mesh2grid_node_aggregate(grid_node_updated)

        # residual
        gx = gx + grid_node_updated

        return gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x


class PredictNet(torch.nn.Module):
    def __init__(self, args):
        super(PredictNet, self).__init__()

        gemb = args.grid_node_embed_dim
        pred_dim = args.grid_node_pred_dim

        # prediction
        self.predict_nn = nn.Sequential(
            nn.Linear(gemb, 256, bias=True),
            nn.SiLU(),
            nn.Linear(256, 64, bias=True),
            nn.SiLU(),
            nn.Linear(64, pred_dim, bias=True)
        )

    def forward(self, gx, mx, me_i, me_x, g2me_i, g2me_x, m2ge_i, m2ge_x):

        # output
        gx = self.predict_nn(gx)

        return gx


def get_graphcast_module(args):
    embed = FeatEmbedding(args)
    gnn_blocks = [
        Grid2MeshEdgeUpdate(args),
        Grid2MeshNodeUpdate(args),
        MeshEdgeUpdate(args),
        MeshNodeUpdate(args),
        Mesh2GridEdgeUpdate(args),
        Mesh2GridNodeUpdate(args),
    ]
    head = PredictNet(args)
    layers = [embed] + gnn_blocks + [head]

    return SequentialModel(*layers)
