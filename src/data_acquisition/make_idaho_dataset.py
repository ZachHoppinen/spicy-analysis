from spicy_snow.retrieval import retrieve_snow_depth

from pathlib import Path

from spicy_snow.IO.user_dates import get_input_dates

from shapely.geometry import box
area = box(-115.574500,43.473139,-113.970496,44.418369)

dates = get_input_dates('2021-05-01')

data_dir = Path('/bsuhome/zacharykeskinen/scratch/idaho_full/')

spicy_ds = retrieve_snow_depth(area = area, dates = dates, work_dir = data_dir, \
                                           job_name = f'idaho', existing_job_name = f'idaho', outfp = data_dir.joinpath('idaho.nc'))