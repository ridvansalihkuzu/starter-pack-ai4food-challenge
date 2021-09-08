import torch
import geopandas as gpd
import rasterio as rio
from rasterio import features
import numpy as np
import os
import zipfile
import tqdm

class PlanetReader(torch.utils.data.Dataset):
    def __init__(self, inputs, labels,label_ids=None, transform=None, min_area_to_ignore = 1000, temp_folder='temp_planet/fields', speed_up=False, selected_time_points=None):
        self.tifs = sorted(inputs)
        self.labels = gpd.read_file(labels)
        self.data_transform = transform
        self.temp_folder=temp_folder
        self.speed_up=speed_up
        self.crop_ids=label_ids

        if label_ids is not None:
            self.crop_ids=label_ids.tolist()

        if selected_time_points is not None:
            self.tifs=np.take(self.tifs, selected_time_points)

        # read coordinate system of tifs and project labels to the same coordinate reference system (crs)
        with rio.open(self.tifs[0]) as image:
            self.crs = image.crs
            print('INFO: Coordinate system of the data is: {}'.format(self.crs))
            self.transform = image.transform

        mask = self.labels.geometry.area > min_area_to_ignore
        print(f"INFO: Ignoring {(~mask).sum()}/{len(mask)} fields with area < {min_area_to_ignore}m2")
        self.labels = self.labels.loc[mask]

        self.labels = self.labels.to_crs(self.crs)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        feature = self.labels.iloc[item]

        npyfile = os.path.join(self.temp_folder, "fid_{}.npz".format(feature.fid))
        if os.path.exists(npyfile): # use saved numpy array if already created
            try:
                object = np.load(npyfile)
                image_stack = object["image_stack"]
                mask = object["mask"]
            except zipfile.BadZipFile:
                print("ERROR: {} is a bad zipfile...".format(npyfile))
                raise
        else:

            left, bottom, right, top = feature.geometry.bounds

            window = rio.windows.from_bounds(left, bottom, right, top, self.transform)

            # reads each tif in tifs on the bounds of the feature. shape T x D x H x W
            image_stack = np.stack([rio.open(tif).read(window=window) for tif in self.tifs])

            # get meta data from first image to get the windowed transform
            with rio.open(self.tifs[0]) as src:
                win_transform = src.window_transform(window)

            out_shape = image_stack[0,0].shape
            assert out_shape[0] > 0 and out_shape[1] > 0, "WARNING: fid:{} image stack shape {} is zero in one dimension".format(feature.fid,image_stack.shape)

            # rasterize polygon to get positions of field within crop
            mask = features.rasterize(feature.geometry, all_touched=True,
                                        transform=win_transform, out_shape=image_stack[0,0].shape)

            if self.speed_up:
                print("INFO: Saving time series to {} for faster loading next time...".format(npyfile))
                # save image stack as zipped numpy arrays for faster loading next time
                os.makedirs(self.temp_folder, exist_ok=True)
                np.savez(npyfile, image_stack=image_stack, mask=mask, feature=feature.drop("geometry").to_dict())

        if self.data_transform is not None:
            image_stack, mask = self.data_transform(image_stack, mask)

        if self.crop_ids is not None:
            label=self.crop_ids.index(feature.crop_id)
        else:
            label=feature.crop_id


        return image_stack, label, mask * feature.fid
