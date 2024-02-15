from pathlib import Path

from spicy_snow.retrieval import retrieve_snow_depth
from spicy_snow.IO.user_dates import get_input_dates

import shapely

# change to your minimum longitude, min lat, max long, max lat
area = shapely.geometry.box(8, 45.5, 12, 48)

# this will be where your results are saved
out_nc = Path('./switzerland.nc').expanduser()

# this will generate a tuple of dates from the previous August 1st to this date
dates = get_input_dates('2018-03-01') # run on all s1 images from (2020-08-01, 2021-04-01) in this example

spicy_ds = retrieve_snow_depth(area = area, dates = dates, 
                               work_dir = Path('~/scratch/switzerland').expanduser(), 
                               job_name = f'switerzerland',
                               existing_job_name = 'switerzerland',
                               debug=False,
                               outfp=out_nc,
                               params = [2, 0.5, 0.44])