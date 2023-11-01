import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

from itertools import product
from tqdm.contrib.itertools import product

import warnings

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_bias(x, y): return np.mean(x - y)

def get_stats(x, y):
    if type(x) == xr.DataArray: x = x.values.ravel()
    if type(y) == xr.DataArray: y = y.values.ravel()
    idx = (~np.isnan(x)) & (~np.isnan(y))
    x, y = x[idx], y[idx]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="An input array is constant")
        r, p = pearsonr(x, y)
    b = get_bias(x, y)
    mae = mean_absolute_error(x, y)
    rmse = mean_squared_error(x, y, squared = False)
    return r, b, mae, rmse

npy_dirs = Path('/bsuhome/zacharykeskinen/scratch/spicy/param_npys')

sds = [np.load(Path(fp).joinpath('lidar.npy')) for fp in npy_dirs.glob('*_*-*-*')]
sds = np.concatenate(sds)

# Create parameter space
A = np.round(np.arange(1, 3.1, 0.1), 2)
B = np.round(np.arange(0, 2.01, 0.1), 2)
C = np.round(np.arange(0, 1.001, 0.01), 2)
locs = [fp.stem for fp in npy_dirs.glob('*_*-*-*') if (fp.is_dir())]

da = xr.DataArray(np.zeros((len(A), len(B), len(C))) , coords = [A, B, C], dims = ['A', 'B', 'C'], name = 'pearsonr')
res = xr.merge([da, da.copy().rename('mae'), da.copy().rename('rmse'), da.copy().rename('bias')])

loc_dirs = list(npy_dirs.glob('*_*-*-*'))
# for a, b, c in product(A, B, C):
#     spicy = [np.load(fp.joinpath(f'{a}_{b}_{c}.npy')) for fp in loc_dirs]
#     spicy = np.concatenate(spicy)
#     r, bias, mae, rmse = get_stats(sds, spicy)
#     for name, var in zip(['pearsonr', 'mae', 'rmse', 'bias'], [r, mae, rmse, bias]):
#         res[name].loc[dict(A = a, B = b, C = c)] = var

# res.to_netcdf(npy_dirs.joinpath('grouped.nc'))

# dry only
sds = []
for fp in loc_dirs:
    wet = np.load(fp.joinpath('wet.npy'))
    sds.append(np.load(Path(fp).joinpath('lidar.npy'))[wet == 0])
sds = np.concatenate(sds)


for a, b, c in product(A, B, C):
    spicy = []
    for fp in loc_dirs:
        wet = np.load(fp.joinpath('wet.npy'))
        spicy.append(np.load(fp.joinpath(f'{a}_{b}_{c}.npy'))[wet == 0])
    spicy = np.concatenate(spicy)
    r, bias, mae, rmse = get_stats(sds, spicy)
    for name, var in zip(['pearsonr', 'mae', 'rmse', 'bias'], [r, mae, rmse, bias]):
        res[name].loc[dict(A = a, B = b, C = c)] = var

res.to_netcdf(npy_dirs.joinpath('dry_grouped.nc'))