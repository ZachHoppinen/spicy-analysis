import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

from tqdm import tqdm
print('starting')

data_dir = Path('~/scratch/West_US_Canada').expanduser()
assert data_dir.exists()
fps = list(data_dir.glob('*.nc'))

b_ds = xr.open_dataset('~/scratch/spicy/SnowEx-Data/Banner_2021-03-15.nc')

dss = []
for fp in tqdm(fps):
    dt = pd.to_datetime(fp.stem.split('_')[1])
    if dt > pd.to_datetime('2019-08-01') and dt < pd.to_datetime('2021-04-01'):
        ds = xr.open_dataset(fp)['snd']
        ds = ds.expand_dims(time = [dt])
        ds = ds.sel(lat = slice(b_ds.y.max(), b_ds.y.min()), lon = slice(b_ds.x.min(), b_ds.x.max()))
        dss.append(ds)
full = xr.concat(dss, dim = 'time')

full.to_netcdf(data_dir.joinpath('combo.nc'))