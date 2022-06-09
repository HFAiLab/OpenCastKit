import hfai_env
hfai_env.set_env("weather")

import hfai
hfai.set_watchdog_time(86400)

import dask
import numpy as np
import xarray as xr
import pickle
from pathlib import Path
from tqdm import tqdm

from ffrecord import FileWriter

np.random.seed(2022)

DATADIR = '/***/hfai'
DATANAMES = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
             'geopotential@1000', 'geopotential@50', 'geopotential@500', 'geopotential@850',
             'mean_sea_level_pressure', 'relative_humidity@500', 'relative_humidity@850',
             'surface_pressure', 'temperature@500', 'temperature@850', 'total_column_water_vapour',
             'u_component_of_wind@1000', 'u_component_of_wind@500', 'u_component_of_wind@850',
             'v_component_of_wind@1000', 'v_component_of_wind@500', 'v_component_of_wind@850']
DATAMAP = {
    'geopotential': 'z',
    'relative_humidity': 'r',
    'temperature': 't',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v'
}


def dataset_to_sample(name, raw_data):
    tmpdata = raw_data
    print(f"{name}-tmpdata.shape: {tmpdata.shape}")

    meanv, maxv, minv, stdv = np.mean(tmpdata.values), np.max(tmpdata.values), np.min(tmpdata.values), np.std(tmpdata.values)
    print(f"Var: {name} | Mean:{meanv:.4f}, Max:{maxv:.4f}, Min:{minv:.4f}, Std:{stdv:.4f}")

    tmpdata = (tmpdata - meanv) / stdv

    dataset_x = tmpdata[0: -2]
    dataset_y_0 = tmpdata[1: -1]
    dataset_y_1 = tmpdata[2:]

    return dataset_x, dataset_y_0, dataset_y_1, meanv, stdv


def write_dataset(data_x, data_y_0, data_y_1, out_file):

    print(f"out_file: {out_file} | x: {data_x.shape}, y_0: {data_y_0.shape}, y_1: {data_y_1.shape}")
    n_sample = data_x.shape[0]

    # 初始化ffrecord
    writer = FileWriter(out_file, n_sample)

    for item in zip(data_x, data_y_0, data_y_1):
        bytes_ = pickle.dumps(item)
        writer.write_one(bytes_)
    writer.close()


def fetch_nc2npy(out_dir):
    # load weather data
    datas = []
    for file in DATANAMES:
        tmp = xr.open_mfdataset((f'{DATADIR}/era5_{file}_*.nc'), combine='by_coords')
        if '@' in file:
            k, v = file.split('@')
            tmp = tmp.rename_vars({DATAMAP[k]: f'{DATAMAP[k]}@{v}'})
        datas.append(tmp)
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        valid_data = xr.merge(datas, compat="identical", join="inner")

    nfiles = 100 # 每多少样本打包成一份

    mean, std = [], []
    for name in ['u10', 'v10', 't2m', 'z@1000', 'z@50', 'z@500', 'z@850', 'msl', 'r@500', 'r@850', 'sp', 't@500', 't@850', 'tcwv', 'u@1000', 'u@500', 'u@850', 'v@1000', 'v@500', 'v@850']:
        raw = valid_data[name]

        # split sample data
        seq_x, seq_y_0, seq_y_1, seq_mean, seq_std = dataset_to_sample(name, raw)
        print(f"{name} | seq_x.shape: {seq_x.shape}, seq_y_0.shape: {seq_y_0.shape}, seq_y_1.shape: {seq_y_1.shape}")

        chunk_id = 1
        for i in tqdm(range(0, seq_x.shape[0], nfiles)):
            start = i
            end = min(i+nfiles, seq_x.shape[0])

            np.save(out_dir / f'{name}_tmp_x_{chunk_id:03d}.npy', seq_x[start:end])
            np.save(out_dir / f'{name}_tmp_y_0_{chunk_id:03d}.npy', seq_y_0[start:end])
            np.save(out_dir / f'{name}_tmp_y_1_{chunk_id:03d}.npy', seq_y_1[start:end])

            chunk_id += 1

        print('\n', end='')

        mean.append(seq_mean)
        std.append(seq_std)

    mean = np.asarray(mean)
    std = np.asarray(std)
    print('save scaler')
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump({
            "mean": mean,
            "std": std
        }, f)


def fetch_npy2dataset(tmp_out_dir, ffr_out_dir):

    n_chunks = 0
    for file in tmp_out_dir.iterdir():
        if file.is_file():
            n_chunks += 1
    n_chunks = int(n_chunks / 20 / 3)

    n_train = int(n_chunks * 0.85)

    for i in range(n_chunks):
        chunk_id = i+1

        sample, label_0, label_1 = [], [], []
        for name in ['u10', 'v10', 't2m', 'z@1000', 'z@50', 'z@500', 'z@850', 'msl', 'r@500', 'r@850', 'sp', 't@500', 't@850', 'tcwv', 'u@1000', 'u@500', 'u@850', 'v@1000', 'v@500', 'v@850']:
            seq_x = np.load(tmp_out_dir / f'{name}_tmp_x_{chunk_id:03d}.npy')
            seq_y_0 = np.load(tmp_out_dir / f'{name}_tmp_y_0_{chunk_id:03d}.npy')
            seq_y_1 = np.load(tmp_out_dir / f'{name}_tmp_y_1_{chunk_id:03d}.npy')

            sample.append(seq_x)
            label_0.append(seq_y_0)
            label_1.append(seq_y_1)

        sample = np.stack(sample, axis=-1)
        label_0 = np.stack(label_0, axis=-1)
        label_1 = np.stack(label_1, axis=-1)
        print(f"sample.shape: {sample.shape}, label_0.shape: {label_0.shape}, label_1.shape: {label_1.shape}\n")

        if i < n_train:
            write_dataset(sample, label_0, label_1, ffr_out_dir / f"train.ffr/{chunk_id:03d}.ffr")
        else:
            write_dataset(sample, label_0, label_1, ffr_out_dir / f"val.ffr/{chunk_id-n_train:03d}.ffr")


def dump_era5(tmp_out_dir, ffr_out_dir):

    tmp_out_dir.mkdir(exist_ok=True, parents=True)
    (ffr_out_dir / "train.ffr").mkdir(exist_ok=True, parents=True)
    (ffr_out_dir / "val.ffr").mkdir(exist_ok=True, parents=True)

    fetch_nc2npy(tmp_out_dir)
    fetch_npy2dataset(tmp_out_dir, ffr_out_dir)

    print("Done.")


if __name__ == "__main__":
    tmp_out_dir = Path("/****/era5_tmp/")
    ffr_out_dir = Path("/***/era5/")

    dump_era5(tmp_out_dir, ffr_out_dir)