import os
os.environ['http_proxy'] = "http://vdi-proxy.high-flyer.cn:3128"
os.environ['https_proxy'] = "http://vdi-proxy.high-flyer.cn:3128"

from datetime import datetime, timedelta
from pathlib import Path
import cdsapi

DATADIR = Path('./output/rawdata/')
all_days = [
    '01','02','03','04','05','06','07','08','09','10','11','12','13','14','15',
    '16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'
]
all_times = [
    '00:00','06:00','12:00','18:00'
]
all_variables = [
    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
    'geopotential@1000', 'geopotential@50', 'geopotential@500', 'geopotential@850',
    'mean_sea_level_pressure', 'relative_humidity@500', 'relative_humidity@850',
    'surface_pressure', 'temperature@500', 'temperature@850', 'total_column_water_vapour',
    'u_component_of_wind@1000', 'u_component_of_wind@500', 'u_component_of_wind@850',
    'v_component_of_wind@1000', 'v_component_of_wind@500', 'v_component_of_wind@850',
    'total_precipitation'
]


def download_single_file(
        variable,
        level_type,
        year,
        pressure_level='1000',
        month='01',
        day='01',
        output_dir=None,
):
    """
    Download a single file from the ERA5 archive.

    :param variable: Name of variable in archive
    :param level_type: 'single' or 'pressure'
    :param year: Year(s) to download data
    :param pressure_level: Pressure levels to download. None for 'single' output type.
    :param month: Month(s) to download data
    :param day: Day(s) to download data
    :param time: Hour(s) to download data. Format: 'hh:mm'
    """

    if level_type == 'pressure':
        var_name = f'{variable}@{pressure_level}'
    else:
        var_name = variable

    if type(day) is list:
        fn = f'era5_{var_name}_{year}_{month}_6_hourly.nc'
    else:
        fn = f'era5_{var_name}_{year}_{month}_{day}_6_hourly.nc'

    c = cdsapi.Client(progress=False)

    request_parameters = {
        'product_type':   'reanalysis',
        'expver':         '1',
        'format':         'netcdf',
        'variable':       variable,
        'year':           year,
        'month':          month,
        'day':            day,
        'time':           all_times,
    }
    request_parameters.update({'pressure_level': pressure_level} if level_type == 'pressure' else {})

    c.retrieve(
        f'reanalysis-era5-{level_type}-levels',
        request_parameters,
        str(output_dir / fn)
    )

    print(f"Saved file: {output_dir / fn}")


def main(
        variable='geopotential',
        level_type='pressure',
        years='1979',
        pressure_level='1000',
        month='01',
        day='01'
):
    """
    Command line script to download single or several files from the ERA5 archive.

    :param variable: Name of variable in archive
    :param level_type: 'single' or 'pressure'
    :param years: Years to download data. Each year is saved separately
    :param pressure_level: Pressure levels to download. None for 'single' output type.
    :param month: Month(s) to download data
    :param day: Day(s) to download data
    :param time: Hour(s) to download data. Format: 'hh:mm'
    """
    # Make sure output directory exists
    output_dir = DATADIR / 'new_files'
    output_dir.mkdir(parents=True, exist_ok=True)

    if level_type == 'pressure':
        assert pressure_level is not None, 'Pressure level must be defined.'

    download_single_file(
        variable=variable,
        level_type=level_type,
        year=years,
        pressure_level=pressure_level,
        month=month,
        day=day,
        output_dir=output_dir
    )


def fetch_a_day(date):
    for var in all_variables:
        if '@' in var:
            level_type= 'pressure'
            var, pressure_level = var.split('@')
        else:
            level_type = 'single'
            pressure_level = ''

        main(
            variable=var,
            level_type=level_type,
            years=f'{date.year}',
            pressure_level=pressure_level,
            month=f'{date.month:02d}',
            day=f'{date.day:02d}'
        )

    # move data
    for item in all_variables:
        os.system(f'mv {DATADIR}/new_files/era5_{item}_* {DATADIR}/era5_6_hourly/{item}/')


def fetch_a_month(date):
    for var in all_variables:
        if '@' in var:
            level_type= 'pressure'
            var, pressure_level = var.split('@')
        else:
            level_type = 'single'
            pressure_level = ''

        main(
            variable=var,
            level_type=level_type,
            years=f'{date.year}',
            pressure_level=pressure_level,
            month=f'{date.month:02d}',
            day=all_days
        )

    # move data
    for item in all_variables:
        os.system(f'mv {DATADIR}/new_files/era5_{item}_* {DATADIR}/era5_6_hourly/{item}/')


if __name__ == '__main__':

    fetch_date = datetime.today() - timedelta(days=6)
    # fetch_date = datetime(2022, 11, 23)

    fetch_a_day(fetch_date)
    # fetch_a_month(fetch_date)
