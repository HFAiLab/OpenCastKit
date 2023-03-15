import dask
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import xarray as xr
import pickle
from pathlib import Path

from ffrecord import FileWriter
from data_factory.graph_tools import fetch_time_features

np.random.seed(2022)

DATADIR = './output/rawdata/era5_6_hourly/'
DATANAMES = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
             'geopotential@1000', 'geopotential@50', 'geopotential@500', 'geopotential@850',
             'mean_sea_level_pressure', 'relative_humidity@500', 'relative_humidity@850',
             'surface_pressure', 'temperature@500', 'temperature@850', 'total_column_water_vapour',
             'u_component_of_wind@1000', 'u_component_of_wind@500', 'u_component_of_wind@850',
             'v_component_of_wind@1000', 'v_component_of_wind@500', 'v_component_of_wind@850',
             'total_precipitation']
DATAMAP = {
    'geopotential': 'z',
    'relative_humidity': 'r',
    'temperature': 't',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v'
}


def dataset_to_sample(raw_data, mean, std):
    tmpdata = (raw_data - mean) / std

    xt0 = tmpdata[:-2]
    xt1 = tmpdata[1:-1]
    yt = tmpdata[2:]

    return xt0, xt1, yt


def write_dataset(x0, x1, y, out_file):
    n_sample = x0.shape[0]

    # 初始化ffrecord
    writer = FileWriter(out_file, n_sample)

    for item in zip(x0, x1, y):
        bytes_ = pickle.dumps(item)
        writer.write_one(bytes_)
    writer.close()


def load_ndf(time_scale):
    datas = []
    for file in DATANAMES:
        tmp = xr.open_mfdataset(f'{DATADIR}/{file}/*.nc', combine='by_coords').sel(time=time_scale)
        if '@' in file:
            k, v = file.split('@')
            tmp = tmp.rename_vars({DATAMAP[k]: f'{DATAMAP[k]}@{v}'})
        datas.append(tmp)
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        valid_data = xr.merge(datas, compat="identical", join="inner")

    return valid_data


def fetch_dataset(cursor_time, out_dir):
    # load weather data

    step = (cursor_time.year - 1979) * 12 + (cursor_time.month - 1) + 1
    start = cursor_time.strftime('%Y-%m-%d %H:%M:%S')
    end = (cursor_time + relativedelta(months=1, hours=7)).strftime('%Y-%m-%d %H:%M:%S')
    print(f'Step {step} | from {start} to {end}')

    time_scale = slice(start, end)

    with open("./output/data/scaler.pkl", "rb") as f:
        pkl = pickle.load(f)
        Mean = pkl["mean"]
        Std = pkl["std"]

    valid_data = load_ndf(time_scale)

    # era5 features
    Xt0, Xt1, Yt = [], [], []
    for i, name in enumerate(['u10', 'v10', 't2m', 'z@1000', 'z@50', 'z@500', 'z@850', 'msl', 'r@500', 'r@850', 'sp', 't@500', 't@850', 'tcwv', 'u@1000', 'u@500', 'u@850', 'v@1000', 'v@500', 'v@850']):
        raw = valid_data[name]

        # split sample data
        xt0, xt1, yt = dataset_to_sample(raw, Mean[i], Std[i])

        Xt0.append(xt0)
        Xt1.append(xt1)
        Yt.append(yt)

    Xt0 = np.stack(Xt0, axis=-1)
    Xt1 = np.stack(Xt1, axis=-1)
    Yt = np.stack(Yt, axis=-1)

    # time-dependent features
    time_features = []
    for i in range(len(valid_data['time'])):
        cursor_time = cursor_time + timedelta(hours=6) * i
        tmp_feats = fetch_time_features(cursor_time)
        time_features.append(tmp_feats)
    time_features = np.asarray(time_features)

    Xt0 = np.concatenate([Xt0[:, 1:], time_features[:-2]], axis=-1)
    Xt1 = np.concatenate([Xt1[:, 1:], time_features[1:-1]], axis=-1)
    Yt = np.concatenate([Yt[:, 1:], time_features[2:]], axis=-1)
    print(f"Xt0.shape: {Xt0.shape}, Xt1.shape: {Xt1.shape}, Yt.shape: {Yt.shape}\n")

    write_dataset(Xt0, Xt1, Yt, out_dir / f"{step:03d}.ffr")


def dump_era5(out_dir):
    out_dir.mkdir(exist_ok=True, parents=True)

    start_time = datetime(1979, 1, 1, 0, 0)
    end_time = datetime(2023, 2, 1, 0, 0)

    cursor_time = start_time
    while True:
        if cursor_time >= end_time:
            break

        fetch_dataset(cursor_time, out_dir)
        cursor_time += relativedelta(months=1)


if __name__ == "__main__":
    out_dir = Path("./output/data/train.ffr")
    dump_era5(out_dir)
