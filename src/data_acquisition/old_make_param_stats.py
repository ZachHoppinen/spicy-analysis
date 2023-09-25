import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from multiprocessing import Pool
from functools import partial

from pathlib import Path
from tqdm import tqdm

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error

from itertools import product
import warnings

# from tqdm.contrib.itertools import product

def get_stats(a, b):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="An input array is constant")
        r, p = pearsonr(a, b)
    error = mae(a, b)
    rmse = mean_squared_error(a, b, squared=False)
    bias = np.mean(a - b)
    return r, error, rmse, bias


def make_stat_da(iter_num, loc_fp):
    
    print(f'Starting {loc_fp.stem}...')

    param_fps = list(param_fp.glob('*'))[0]
    stems= [f.stem for f in list(param_fps.glob('*_*_*.npy'))]

    A = np.unique([float(s.split('_')[0]) for s in stems])
    B = np.unique([float(s.split('_')[1]) for s in stems])
    C = np.unique([float(s.split('_')[2]) for s in stems])
    
    iterations = np.arange(iter_num)

    res = np.zeros((1, len(A), len(B), len(C), len(iterations)))

    da = xr.DataArray(res, coords = [[loc_fp.stem], A, B, C, iterations], dims = ['location', 'A', 'B','C', 'iteration'], name = 'pearsonr')
    res_ds = xr.merge([da, da.copy().rename('mae'), da.copy().rename('rmse'), da.copy().rename('bias')])

    lidar_orig = np.load(loc_fp.joinpath('lidar.npy'))
    elev = np.load(loc_fp.joinpath('elev.npy'))
    trees = np.load(loc_fp.joinpath('trees.npy'))

    idx = (trees <= 1) & (elev > 0)

    if 'Cottonwood' in loc_fp.stem:
        print(f"Size of datarray: {res_ds['pearsonr'].data.shape}")
        print(f"Number of iterations: {iter_num}")
        print(f'Used fraction: {np.sum(idx)/elev.size}')

    for a, b, c in product(A, B, C):
        sds_orig = np.load(loc_fp.joinpath(f'{a}_{b}_{c}.npy'))
        combo = np.vstack([lidar_orig, sds_orig])
        combo = combo.T[idx].T
        for iter in iterations:
            if iter != 0:
                id_iter = np.random.choice(combo.shape[1], combo.shape[1], replace = True)
                sds, lidar = combo.T[id_iter].T
            else:
                sds, lidar = combo
            r, mean_error, rmse, bias = get_stats(lidar, sds)
            res_ds['pearsonr'].loc[dict(location = loc_fp.stem, A = a, B = b, C = c, iteration = iter)] = r
            res_ds['mae'].loc[dict(location = loc_fp.stem, A = a, B = b, C = c, iteration = iter)] = mean_error
            res_ds['rmse'].loc[dict(location = loc_fp.stem, A = a, B = b, C = c, iteration = iter)] = rmse
            res_ds['bias'].loc[dict(location = loc_fp.stem, A = a, B = b, C = c, iteration = iter)] = bias
    
    print(f'Finishing {loc_fp.stem}!')
    for dv in res_ds.data_vars:
        res_ds[dv] = res_ds[dv].astype(float)
    res_ds.to_netcdf(loc_fp.parent.joinpath('stats_ncs', loc_fp.stem + '_stats.nc'))


out_fp = Path('/bsuhome/zacharykeskinen/scratch/spicy/param_stats_all_trees.nc')

if out_fp.exists():
    print('Already exists...')
    # res_ds = xr.load_dataset(out_fp)


param_fp = Path('/bsuhome/zacharykeskinen/scratch/spicy/param_npys')

param_fp.joinpath('stats_ncs').mkdir(exist_ok = True)

locs = list(param_fp.glob('*'))
locs = [l.stem for l in locs]

pool = Pool()
    
number_of_iterations = 100
pool.map(partial(make_stat_da, number_of_iterations), param_fp.glob('*_*-*-*'))

das = [xr.open_dataset(fp) for fp in param_fp.joinpath('stats_ncs').glob('*.nc')]
ds = xr.merge(das)

ds.to_netcdf(param_fp.joinpath('param_stats.nc'))