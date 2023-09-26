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


def make_stat_da(tree_idx_low, tree_idx_high, loc_fp):
    
    print(f'Starting {loc_fp.stem}...')

    stems= [f.stem for f in list(loc_fp.glob('*_*_*.npy'))]

    A = np.unique([float(s.split('_')[0]) for s in stems])
    B = np.unique([float(s.split('_')[1]) for s in stems])
    C = np.unique([float(s.split('_')[2]) for s in stems])

    res = np.zeros((1, len(A), len(B), len(C)))

    da = xr.DataArray(res, coords = [[loc_fp.stem], A, B, C], dims = ['location', 'A', 'B','C'], name = 'pearsonr')
    res_ds = xr.merge([da, da.copy().rename('mae'), da.copy().rename('rmse'), da.copy().rename('bias')])

    lidar_orig = np.load(loc_fp.joinpath('lidar.npy'))
    elev = np.load(loc_fp.joinpath('elev.npy'))
    trees = np.load(loc_fp.joinpath('trees.npy'))

    idx = (trees <= tree_idx_high) & (elev > 0) & (trees >= tree_idx_low)

    if 'Cottonwood' in loc_fp.stem:
        print(f"Size of datarray: {res_ds['pearsonr'].data.shape}")
        print(f'Used fraction: {np.sum(idx)/elev.size}')

    for a, b, c in product(A, B, C):
        sds_orig = np.load(loc_fp.joinpath(f'{a}_{b}_{c}.npy'))
        combo = np.vstack([lidar_orig, sds_orig])
        combo = combo.T[idx].T

        sds, lidar = combo
        r, mean_error, rmse, bias = get_stats(lidar, sds)
        res_ds['pearsonr'].loc[dict(location = loc_fp.stem, A = a, B = b, C = c)] = r
        res_ds['mae'].loc[dict(location = loc_fp.stem, A = a, B = b, C = c)] = mean_error
        res_ds['rmse'].loc[dict(location = loc_fp.stem, A = a, B = b, C = c)] = rmse
        res_ds['bias'].loc[dict(location = loc_fp.stem, A = a, B = b, C = c)] = bias
    
    for dv in res_ds.data_vars:
        res_ds[dv] = res_ds[dv].astype(float)
    res_ds.to_netcdf(loc_fp.parent.joinpath('stats_ncs', loc_fp.stem + f'_{tree_idx_low:.2f}_{tree_idx_high:.2f}_stats.nc'))

    print(f'Finishing {loc_fp.stem}!')


param_fp = Path('/bsuhome/zacharykeskinen/scratch/spicy/param_npys')

param_fp.joinpath('stats_ncs').mkdir(exist_ok = True)


out_fp = Path('/bsuhome/zacharykeskinen/scratch/spicy/param_stats.nc')

if out_fp.exists():
    print('Already exists...')

pool = Pool()
    
pool.map(partial(make_stat_da, 0, 1), param_fp.glob('*_*-*-*'))
das = [xr.open_dataset(fp) for fp in param_fp.joinpath('stats_ncs').glob('*0.00_1.00_stats.nc')]
ds = xr.merge(das)
ds.to_netcdf(param_fp.joinpath('param_stats_all.nc'))

pool = Pool()
pool.map(partial(make_stat_da, 0, 0.25), param_fp.glob('*_*-*-*'))
das = [xr.open_dataset(fp) for fp in param_fp.joinpath('stats_ncs').glob('*0.00_0.25_stats.nc')]
ds = xr.merge(das)
ds.to_netcdf(param_fp.joinpath('param_stats_low_fcf.nc'))

pool = Pool()
pool.map(partial(make_stat_da, 0.75, 1), param_fp.glob('*_*-*-*'))
das = [xr.open_dataset(fp) for fp in param_fp.joinpath('stats_ncs').glob('*0.75_1.00_stats.nc')]
ds = xr.merge(das)
ds.to_netcdf(param_fp.joinpath('param_stats_high_fcf.nc'))

# locs = list(param_fp.glob('*'))
# locs = [l.stem for l in locs]