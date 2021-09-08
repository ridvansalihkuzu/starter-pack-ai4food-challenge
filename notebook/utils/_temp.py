# Built-in modules
import os
import glob
import json
from typing import Tuple, List
from datetime import datetime, timedelta

# Basics of Python data handling and visualization
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from tqdm.auto import tqdm

# Data reding for training validation purposes:
from planet_reader import PlanetReader



# Imports from eo-learn and sentinelhub-py
from sentinelhub import CRS, BBox, SHConfig, DataCollection

from eolearn.core import (FeatureType,
                          EOPatch,
                          EOTask,
                          LinearWorkflow,
                          EOExecutor,
                          LoadTask,
                          SaveTask)
from eolearn.io import GeoDBVectorImportTask, SentinelHubInputTask
from eolearn.geometry import VectorToRaster

# Visualisation utils
from notebook.utils.utils import get_extent, md5_encode_files

# TODO: CHANGE THIS AFTER HAVING DOWNLOAD LINK!!!
brandenburg_gt_dir='/mnt/ushelf/datasets/ai4food_challenge/brandenburg-gt/brandenburg_crops_train_2018.geojson'
brandenburg_planet_dir= '/notebook/data/planet/UTM-24000/33N'



#CHECK TRAINING DATA
brandenburg_planet_train_files=glob.glob(brandenburg_planet_dir + '/**/*.tif', recursive=True)
brandenburg_sentinel_1_train_files=glob.glob(brandenburg_planet_dir + '/**/*.tif', recursive=True)
brandenburg_sentinel_2_train_files=glob.glob(brandenburg_planet_dir + '/**/*.tif', recursive=True)

#CHECK TARGET (LABEL) DATA FORMAT:
brandenburg_gt=gpd.read_file(brandenburg_gt_dir)
print('INFO: Number of fields: {}\n'.format(len(brandenburg_gt)))
brandenburg_gt.info()
brandenburg_gt.head()

#CHECK LABEL IDs AND LABEL NAMES:

label_ids=brandenburg_gt['crop_id'].unique()
label_names=brandenburg_gt['crop_name'].unique()

print('INFO: Label IDs: {}'.format(label_ids))
print('INFO: Label Names: {}'.format(label_names))


#VISUALISE SOME OF THE FIELDS FROM PLANET DATA:
planet_reader = PlanetReader(brandenburg_planet_train_files,brandenburg_gt_dir)

selected_day_of_year = [140,185,210,250]
for crop_id, crop_name in zip(label_ids,label_names):
    X,y,fid = next(iter(planet_reader))
    print(y)
    break
