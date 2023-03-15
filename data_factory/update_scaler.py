import hfai_env
hfai_env.set_env("weather")

import xarray as xr
import pickle
import numpy as np

DATADIR = './output/rawdata/era5_6_hourly/'
DATANAMES = {
    '10m_u_component_of_wind': 'u10',
    '10m_v_component_of_wind': 'v10',
    '2m_temperature': 't2m',
    'geopotential@1000': 'z',
    'geopotential@50': 'z',
    'geopotential@500': 'z',
    'geopotential@850': 'z',
    'mean_sea_level_pressure': 'msl',
    'relative_humidity@500': 'r',
    'relative_humidity@850': 'r',
    'surface_pressure': 'sp',
    'temperature@500': 't',
    'temperature@850': 't',
    'total_column_water_vapour': 'tcwv',
    'u_component_of_wind@1000': 'u',
    'u_component_of_wind@500': 'u',
    'u_component_of_wind@850': 'u',
    'v_component_of_wind@1000': 'v',
    'v_component_of_wind@500': 'v',
    'v_component_of_wind@850': 'v',
    'total_precipitation': 'tp'}
DATAVARS = ['u10', 'v10', 't2m', 'z@1000', 'z@50', 'z@500', 'z@850', 'msl', 'r@500', 'r@850', 'sp', 't@500', 't@850',
            'tcwv', 'u@1000', 'u@500', 'u@850', 'v@1000', 'v@500', 'v@850', 'tp']


if __name__ == '__main__':

    Mean, Std = [], []

    for k, v in DATANAMES.items():

        raw = xr.open_mfdataset((f'{DATADIR}/{k}/*.nc'), combine='by_coords')

        if k == 'total_precipitation':
            data = raw[v].values[:, :, :, 0] * 1e9
            np.nan_to_num(data, copy=False)
            meanv = np.mean(data) / 1e9
            stdv = np.std(data) / 1e9
        else:
            data = raw[v].values
            np.nan_to_num(data, copy=False)
            meanv = np.mean(data)
            stdv = np.std(data)

        print(f'Var: {k} | mean: {meanv}, std: {stdv}')

        Mean.append(meanv)
        Std.append(stdv)

    with open("./output/data/scaler.pkl", "wb") as f:
        pickle.dump({
            "mean": np.asarray(Mean),
            "std": np.asarray(Std)
        }, f)