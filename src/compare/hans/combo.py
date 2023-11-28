import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

from tqdm import tqdm
print('starting')

in_dir = Path('~/scratch/spicy/SnowEx-Data/').expanduser().resolve()
in_dir = Path('~/spicy-snow/SnowEx-Data-Hans/').expanduser().resolve()

out_dir = Path('~/scratch/spicy/SnowEx-Hans-s1').expanduser()

dss = {fp.stem: xr.open_dataset(fp) for fp in in_dir.glob('*.nc')}

data_dir = Path('~/scratch/West_US_Canada').expanduser()
# assert data_dir.exists()
fps = list(data_dir.glob('snd_*.nc'))

for stem, old_ds in dss.items():
    new_dss = []
    print(stem)
    for fp in tqdm(fps):
        dt = pd.to_datetime(fp.stem.split('_')[1])
        if dt > old_ds.time[0] and dt < old_ds.time[-1]:
            ds = xr.open_dataset(fp)['snd']
            ds = ds.expand_dims(time = [dt])
            ds = ds.sel(lat = slice(old_ds.y.max(), old_ds.y.min()), lon = slice(old_ds.x.min(), old_ds.x.max()))
            new_dss.append(ds)

    full = xr.concat(new_dss, dim = 'time')
    full = full.dropna(dim ='time', how ='all').sortby('time')
    full = full.rename({'lat':'y'})
    full = full.rename({'lon':'x'})

    full.to_netcdf(out_dir.joinpath(stem+'.nc'))