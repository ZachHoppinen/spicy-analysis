# FOR GENERATING THE PARAMETER DATASETS! #

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from datetime import datetime

from spicy_snow.processing.snow_index import calc_delta_cross_ratio, calc_delta_gamma, \
    clip_delta_gamma_outlier, calc_snow_index, calc_snow_index_to_snow_depth
from spicy_snow.processing.wet_snow import id_newly_wet_snow, id_wet_negative_si, \
    id_newly_frozen_snow, flag_wet_snow

def make_param_files(data_directory, idx, closest_ts, B, C, dataset, a):
    ds = dataset

    ds = calc_delta_cross_ratio(ds, A = a)
    for b in B:
        ds = calc_delta_gamma(ds, B = b, inplace=False)
        ds = clip_delta_gamma_outlier(ds)
        ds = calc_snow_index(ds)
        ds = id_newly_wet_snow(ds)
        ds = id_wet_negative_si(ds)
        ds = id_newly_frozen_snow(ds)
        ds = flag_wet_snow(ds)
        for c in C:
            ds = calc_snow_index_to_snow_depth(ds, C = c)

            sub = ds.sel(time = closest_ts)
            spicy_sd = sub['snow_depth'].values.ravel()[idx]
            # print(f'Saving {a}_{b}_{c}.npy')
            np.save(data_directory.joinpath(f'{a}_{b}_{c}.npy'), spicy_sd)

# Create parameter space
A = np.round(np.arange(1, 3.1, 0.1), 2)
B = np.round(np.arange(0, 2.01, 0.1), 2)
C = np.round(np.arange(0, 1.001, 0.01), 2)

spicy_netcdfs = list(Path('/bsuhome/zacharykeskinen/scratch/spicy/SnowEx-Data/').glob('*.nc'))

param_dir = Path('~/scratch/spicy/param_npys').expanduser()

param_dir.mkdir(exist_ok = True)

for spicy_nc in spicy_netcdfs:
    # get dataset
    ds_name = spicy_nc.name.split('stacks/')[-1].split('.')[0]
    print(datetime.now(), f' -- starting {ds_name}')
    ds_ = xr.open_dataset(spicy_nc).load()
    dataset = ds_[['s1','deltaVV','ims','fcf', 'lidar-sd', 'lidar-dem', 'snow_depth']]

    # find closest timestep to lidar
    td = abs(pd.to_datetime(dataset.time) - pd.to_datetime(dataset.attrs['lidar-flight-time']))
    closest_ts = dataset.time[np.argmin(td)]

    if 'Frasier_2020-02-11' in spicy_nc.name:
        closest_ts = '2020-02-16T13:09:43.000000000'

    ds_dir = param_dir.joinpath(ds_name)
    
    ds_dir.mkdir(exist_ok = True)

    trees = dataset['fcf'].values.ravel()
    elev = dataset['lidar-dem'].values.ravel()
    lidar = dataset['lidar-sd'].values.ravel()
    spicy_sd = dataset['snow_depth'].sel(time = closest_ts).values.ravel()
    wet = dataset['wet_snow'].sel(time = closest_ts).values.ravel()  # haven't tested wet saving so this may break in future

    idx = (~np.isnan(trees)) & (~np.isnan(elev)) & (~np.isnan(lidar)) & (~np.isnan(spicy_sd)) & (~np.isnan(wet)) # haven't tested wet saving so this may break in future
    # save auxillary files
    np.save(ds_dir.joinpath(f'lidar.npy'), lidar[idx])
    np.save(ds_dir.joinpath(f'trees.npy'), trees[idx])
    np.save(ds_dir.joinpath(f'elev.npy'), elev[idx])
    np.save(ds_dir.joinpath(f'wet.npy'), wet[idx])  # haven't tested wet saving so this may break in future


    # Brute-force processing loop
    pool = Pool()
        
    pool.map(partial(make_param_files, ds_dir, idx, closest_ts, B, C, dataset), A)