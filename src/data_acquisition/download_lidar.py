from pathlib import Path

from spicy_snow.download.snowex_lidar import download_dem, download_snow_depth,\
      download_veg_height, make_site_ds

lidar_dir = Path('/bsuhome/zacharykeskinen/scratch/spicy/lidar')
lidar_dir.mkdir(exist_ok = True)
download_snow_depth(lidar_dir)
download_veg_height(lidar_dir)
download_dem(lidar_dir)
