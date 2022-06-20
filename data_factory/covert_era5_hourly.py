import hfai_env
hfai_env.set_env("weather")

import hfai
hfai.set_watchdog_time(86400)

import dask
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import xarray as xr
import pickle
from pathlib import Path


from ffrecord import FileWriter

np.random.seed(2022)

DATADIR = '/***/era5'
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

    dataset_xt = tmpdata[0: -2]
    dataset_xt1 = tmpdata[1: -1]
    dataset_xt2 = tmpdata[2:]

    return dataset_xt, dataset_xt1, dataset_xt2


def write_dataset(x, xt1, xt2, pt1, out_file):

    n_sample = x.shape[0]

    # 初始化ffrecord
    writer = FileWriter(out_file, n_sample)

    for item in zip(x, xt1, xt2, pt1):
        bytes_ = pickle.dumps(item)
        writer.write_one(bytes_)
    writer.close()


def load_ndf(time_scale):
    datas = []
    for file in DATANAMES:
        tmp = xr.open_mfdataset((f'{DATADIR}/{file}/*.nc'), combine='by_coords').sel(time=time_scale)
        if '@' in file:
            k, v = file.split('@')
            tmp = tmp.rename_vars({DATAMAP[k]: f'{DATAMAP[k]}@{v}'})
        datas.append(tmp)
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        valid_data = xr.merge(datas, compat="identical", join="inner")

    return valid_data


def fetch_dataset(step, out_dir, time_scale):
    # load weather data

    with open("/3fs-jd/prod/private/hwj/era5/scaler.pkl", "rb") as f:
        pkl = pickle.load(f)
        Mean = pkl["mean"]
        Std = pkl["std"]

    valid_data = load_ndf(time_scale)

    Xt, Xt1, Xt2, Pt1 = [], [], [], []
    for i, name in enumerate(['u10', 'v10', 't2m', 'z@1000', 'z@50', 'z@500', 'z@850', 'msl', 'r@500', 'r@850', 'sp', 't@500', 't@850', 'tcwv', 'u@1000', 'u@500', 'u@850', 'v@1000', 'v@500', 'v@850']):
        raw = valid_data[name]

        # split sample data
        xt, xt1, xt2 = dataset_to_sample(raw, Mean[i], Std[i])
        print(f"{name} | xt.shape: {xt.shape}, xt1.shape: {xt1.shape}, xt2.shape: {xt2.shape}, mean: {Mean[i]}, std: {Std[i]}")

        Xt.append(xt)
        Xt1.append(xt1)
        Xt2.append(xt2)

    tpdata = np.nan_to_num(valid_data['tp'].values[:, :, :, 0])
    tpdata = (tpdata - Mean[-1]) / Std[-1]
    Pt1 = tpdata[1: -1]

    Xt = np.stack(Xt, axis=-1)
    Xt1 = np.stack(Xt1, axis=-1)
    Xt2 = np.stack(Xt2, axis=-1)
    print(f"Xt.shape: {Xt.shape}, Xt1.shape: {Xt1.shape}, Xt2.shape: {Xt2.shape}, Pt1.shape: {Pt1.shape}\n")

    write_dataset(Xt, Xt1, Xt2, Pt1, out_dir / f"{step:03d}.ffr")


def dump_era5(out_dir):
    out_dir.mkdir(exist_ok=True, parents=True)

    start_time = datetime.date(1979, 1, 1)
    end_time = datetime.date(2022, 1, 1)

    cursor_time = start_time
    while True:
        if cursor_time >= end_time:
            break

        step = (cursor_time.year - 1979) * 12 + (cursor_time.month - 1) + 1
        start = cursor_time.strftime('%Y-%m-%d %H:%M:%S')
        end = (cursor_time + relativedelta(months=1, hours=7)).strftime('%Y-%m-%d %H:%M:%S')

        print(f'Step {step} | from {start} to {end}')
        fetch_dataset(step, out_dir, slice(start, end))
        cursor_time +=  relativedelta(months=1)


if __name__ == "__main__":
    out_dir = Path("***/era5/data.ffr")
    dump_era5(out_dir)
