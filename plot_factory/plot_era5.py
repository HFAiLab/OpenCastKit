import hfai_env
hfai_env.set_env("weather")

from pathlib import Path
import json
import dask
import dask.array as da
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import xarray as xr
import xarray.ufuncs as xu
import matplotlib.pyplot as plt
from data_factory.dataset import StandardScaler
from functools import partial

from model.afnonet import AFNONet
from utils.tools import load_model
import cartopy.crs as ccrs

DATADIR = '/***/hfai'
SAMPLEDIR = '/***/era5'
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
SAVE_PATH = Path('./output/fourcastnet/')


def imcol(data, img_path, datetime, metric, type='real', **kwargs):
    fig = plt.figure(figsize=(40, 20))
    ax = plt.axes(projection=ccrs.PlateCarree())

    I = data.plot(ax=ax, transform=ccrs.PlateCarree(), add_colorbar=False, add_labels=False, rasterized=True, **kwargs)
    ax.coastlines(resolution='10m')

    dirname = f'{img_path.absolute()}/{metric}_{datetime}_{type}.jpg'

    plt.axis('off')
    plt.savefig(dirname, bbox_inches='tight', pad_inches=0.)
    plt.close(fig)

    return dirname


def plot(valid_data, pred_data, save_path):
    cmap_t = 'RdYlBu_r'

    img_path = save_path / 'pic'
    img_path.mkdir(parents=True, exist_ok=True)
    img_info = {}

    for i in range(len(valid_data.time)):
        real_u = valid_data['u10'].isel(time=i)
        real_v = valid_data['v10'].isel(time=i)
        real_wind = xu.sqrt(real_u ** 2 + real_v ** 2)
        rwmin, rwmax = real_wind.values.min(), real_wind.values.max()

        real_precip = valid_data['tcwv'].isel(time=i)
        rpmin, rpmax = real_precip.values.min(), real_precip.values.max()

        pred_u = pred_data['u10'].isel(time=i)
        pred_v = pred_data['v10'].isel(time=i)
        pred_wind = xu.sqrt(pred_u ** 2 + pred_v ** 2)
        pwmin, pwmax = pred_wind.values.min(), pred_wind.values.max()

        pred_precip = pred_data['tcwv'].isel(time=i)
        ppmin, ppmax = pred_precip.values.min(), pred_precip.values.max()

        datetime = pd.to_datetime(str(real_wind['time'].values))
        datetime = datetime.strftime('%Y-%m-%d %H:%M:%S')
        print(datetime)

        img_info[datetime] = {
            'wind': {
                'real': '',
                'pred': '',
            },
            'precipitation': {
                'real': '',
                'pred': '',
            },
        }

        img_info[datetime]['wind']['real'] = imcol(real_wind, img_path, datetime, metric='wind', type='real', cmap=cmap_t, vmin=min(rwmin, pwmin), vmax=max(rwmax, pwmax))
        img_info[datetime]['wind']['pred'] = imcol(pred_wind, img_path, datetime, metric='wind', type='pred', cmap=cmap_t, vmin=min(rwmin, pwmin), vmax=max(rwmax, pwmax))
        img_info[datetime]['precipitation']['real'] = imcol(real_precip, img_path, datetime, metric='precipitation', type='real', cmap=cmap_t, vmin=min(rpmin, ppmin), vmax=max(rpmax, ppmax))
        img_info[datetime]['precipitation']['pred'] = imcol(pred_precip, img_path, datetime, metric='precipitation', type='pred', cmap=cmap_t, vmin=min(rpmin, ppmin), vmax=max(rpmax, ppmax))

    with open(f'{save_path}/img_info.json', "w") as fp:
        json.dump(img_info, fp)


def get_pred(sample, scaler, times=None, latitude=None, longitude=None):
    # input size
    h, w = 720, 1440
    x_c, y_c, p_c = 20, 20, 1

    data = torch.from_numpy(sample)

    backbone_model = AFNONet(img_size=[h, w], in_chans=x_c, out_chans=y_c, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    load_model(backbone_model, path=SAVE_PATH / 'backbone.pt', only_model=True)
    backbone_model = backbone_model
    criterion = nn.MSELoss()

    backbone_model.eval()
    pred = [data[0].detach().numpy() * scaler.std + scaler.mean]
    x = data[0].unsqueeze(0).transpose(3, 2).transpose(2, 1)
    for i in range(1, data.size(0)):
        x = backbone_model(x)
        tmp = x.transpose(1, 2).transpose(2, 3)

        y = data[i].unsqueeze(0)
        loss = criterion(tmp, y)
        print(f"Step {i}, Loss: {loss}")

        tmp = tmp.detach().numpy()[0] * scaler.std + scaler.mean
        pred.append(tmp)

    pred = np.asarray(pred)

    pred_data = xr.Dataset({
        'u10': (['time', 'latitude', 'longitude'], da.from_array(pred[:, :, :, 0], chunks=(7, 720, 1440))),
        'v10': (['time', 'latitude', 'longitude'], da.from_array(pred[:, :, :, 1], chunks=(7, 720, 1440))),
        'tcwv': (['time', 'latitude', 'longitude'], da.from_array(pred[:, :, :, 13], chunks=(7, 720, 1440))),
    },
        coords={'time': (['time'], times),
                'latitude': (['latitude'], latitude),
                'longitude': (['longitude'], longitude)
                }
    )

    return pred_data


def get_data(start_time, end_time):
    times = slice(start_time, end_time)

    scaler = StandardScaler()
    scaler.load(f'{SAMPLEDIR}/scaler.pkl')

    # load weather data
    datas = []
    for file in DATANAMES:
        tmp = xr.open_mfdataset((f'{DATADIR}/era5_{file}_*.nc'), combine='by_coords').sel(time=times)
        if '@' in file:
            k, v = file.split('@')
            tmp = tmp.rename_vars({DATAMAP[k]: f'{DATAMAP[k]}@{v}'})
        datas.append(tmp)
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        raw_data = xr.merge(datas, compat="identical", join="inner")

    data = []
    for name in ['u10', 'v10', 't2m', 'z@1000', 'z@50', 'z@500', 'z@850', 'msl', 'r@500', 'r@850', 'sp', 't@500', 't@850', 'tcwv', 'u@1000', 'u@500', 'u@850', 'v@1000', 'v@500', 'v@850']:
        raw = raw_data[name].values
        data.append(raw)
    data = np.stack(data, axis=-1)
    data = (data - scaler.mean) / scaler.std
    data = data[:, 1:, :, :]

    return raw_data, data, scaler


if __name__ == '__main__':

    start_time = "2018-09-14 00:00:00"
    end_time = (pd.to_datetime(start_time) + pd.Timedelta("24:00:00") * 3).strftime('%Y-%m-%d %H:%M:%S')

    valid, sample, scaler = get_data(start_time, end_time)
    print(f'sample size: {sample.shape}')
    pred = get_pred(sample, scaler=scaler, times=valid.time, latitude=valid.latitude[1:], longitude=valid.longitude)

    save_path = Path(f'/***/era5_output/{start_time}')
    save_path.mkdir(parents=True, exist_ok=True)

    valid.to_netcdf(f'{save_path}/valid_data.nc')
    pred.to_netcdf(f'{save_path}/pred_data.nc')
    # valid = xr.open_mfdataset(f'{save_path}/valid_data.nc')
    # pred = xr.open_mfdataset(f'{save_path}/pred_data.nc')

    plot(valid, pred, save_path)