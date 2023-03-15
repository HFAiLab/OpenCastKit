import pickle
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from ffrecord import FileReader
from ffrecord.torch import Dataset, DataLoader
import data_factory.graph_tools as gg


class StandardScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def load(self, scaler_dir):
        with open(scaler_dir, "rb") as f:
            pkl = pickle.load(f)
            self.mean = pkl["mean"]
            self.std = pkl["std"]

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean


class ERA5(Dataset):

    def __init__(self, split: str, check_data: bool = True, modelname: str = 'fourcastnet') -> None:

        self.data_dir = Path("./output/data/")

        assert split in ["train", "val"]
        assert modelname in ["fourcastnet", "graphcast"]
        self.split = split
        self.modelname = modelname
        self.fname = str(self.data_dir / f"{split}.ffr")
        self.reader = FileReader(self.fname, check_data)
        self.scaler = StandardScaler()
        self.scaler.load("./output/data/scaler.pkl")

        if self.modelname == 'graphcast':
            self.constant_features = gg.fetch_constant_features()
        else:
            self.constant_features = None

    def __len__(self):
        return self.reader.n

    def __getitem__(self, indices):
        seqs_bytes = self.reader.read(indices)
        samples = []
        for bytes_ in seqs_bytes:
            x0, x1, y = pickle.loads(bytes_)

            if self.modelname == 'fourcastnet':
                x0 = np.nan_to_num(x0[:, :, :-2])
                x1 = np.nan_to_num(x1[:, :, :-2])
                y = np.nan_to_num(y[:, :, :-2])
                samples.append((x0, x1, y))
            else:
                x = np.nan_to_num(np.reshape(np.concatenate([x0, x1, y[:, :, -2:]], axis=-1), [-1, 49]))
                y = np.nan_to_num(np.reshape(y[:, :, :-2], [-1, 20]))
                samples.append((x, y))
        return samples

    def get_scaler(self):
        return self.scaler

    def loader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self, *args, **kwargs)


class EarthGraph(object):
    def __init__(self):
        self.mesh_data = None
        self.grid2mesh_data = None
        self.mesh2grid_data = None

    def generate_graph(self):
        mesh_nodes = gg.fetch_mesh_nodes()

        mesh_6_edges, mesh_6_edges_attrs = gg.fetch_mesh_edges(6)
        mesh_5_edges, mesh_5_edges_attrs = gg.fetch_mesh_edges(5)
        mesh_4_edges, mesh_4_edges_attrs = gg.fetch_mesh_edges(4)
        mesh_3_edges, mesh_3_edges_attrs = gg.fetch_mesh_edges(3)
        mesh_2_edges, mesh_2_edges_attrs = gg.fetch_mesh_edges(2)
        mesh_1_edges, mesh_1_edges_attrs = gg.fetch_mesh_edges(1)
        mesh_0_edges, mesh_0_edges_attrs = gg.fetch_mesh_edges(0)

        mesh_edges = mesh_6_edges + mesh_5_edges + mesh_4_edges + mesh_3_edges + mesh_2_edges + mesh_1_edges + mesh_0_edges
        mesh_edges_attrs = mesh_6_edges_attrs + mesh_5_edges_attrs + mesh_4_edges_attrs + mesh_3_edges_attrs + mesh_2_edges_attrs + mesh_1_edges_attrs + mesh_0_edges_attrs

        self.mesh_data = Data(x=torch.tensor(mesh_nodes, dtype=torch.float),
                              edge_index=torch.tensor(mesh_edges, dtype=torch.long).T.contiguous(),
                              edge_attr=torch.tensor(mesh_edges_attrs, dtype=torch.float))

        grid2mesh_edges, grid2mesh_edge_attrs = gg.fetch_grid2mesh_edges()
        self.grid2mesh_data = Data(x=None,
                                   edge_index=torch.tensor(grid2mesh_edges, dtype=torch.long).T.contiguous(),
                                   edge_attr=torch.tensor(grid2mesh_edge_attrs, dtype=torch.float))

        mesh2grid_edges, mesh2grid_edge_attrs = gg.fetch_mesh2grid_edges()
        self.mesh2grid_data = Data(x=None,
                                   edge_index=torch.tensor(mesh2grid_edges, dtype=torch.long).T.contiguous(),
                                   edge_attr=torch.tensor(mesh2grid_edge_attrs, dtype=torch.float))