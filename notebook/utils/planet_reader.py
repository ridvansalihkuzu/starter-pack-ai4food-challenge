import torch
import geopandas as gpd
import rasterio as rio
from rasterio import features
import numpy as np
import os
import zipfile
import glob
from tqdm import tqdm

class PlanetReader(torch.utils.data.Dataset):
    def __init__(self, input_dir, label_dir, label_ids=None, transform=None, min_area_to_ignore = 1000,  selected_time_points=None):

        self.data_transform = transform
        self.selected_time_points = selected_time_points
        self.crop_ids=label_ids
        if label_ids is not None:
            self.crop_ids=label_ids.tolist()

        self.npyfolder = os.path.abspath(input_dir + "time_series")
        self.labels = PlanetReader._setup(input_dir, label_dir, self.npyfolder, min_area_to_ignore)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        feature = self.labels.iloc[item]

        npyfile = os.path.join(self.npyfolder, "fid_{}.npz".format(feature.fid))
        if os.path.exists(npyfile): # use saved numpy array if already created
            try:
                object = np.load(npyfile)
                image_stack = object["image_stack"]
                mask = object["mask"]
            except zipfile.BadZipFile:
                print("ERROR: {} is a bad zipfile...".format(npyfile))
                raise
        else:
            print("ERROR: {} is a missing...".format(npyfile))
            raise

        if self.data_transform is not None:
            image_stack, mask = self.data_transform(image_stack, mask)

        if self.selected_time_points is not None:
            image_stack = image_stack[self.selected_time_points]

        if self.crop_ids is not None:
            label = self.crop_ids.index(feature.crop_id)
        else:
            label = feature.crop_id

        return image_stack*mask, label, mask


    @staticmethod
    def _setup(input_dir, label_dir, npyfolder, min_area_to_ignore=1000):
        inputs = glob.glob(input_dir + '*.tif', recursive=True)
        tifs = sorted(inputs)
        labels = gpd.read_file(label_dir)

        # read coordinate system of tifs and project labels to the same coordinate reference system (crs)
        with rio.open(tifs[0]) as image:
            crs = image.crs
            print('INFO: Coordinate system of the data is: {}'.format(crs))
            transform = image.transform

        mask = labels.geometry.area > min_area_to_ignore
        print(f"INFO: Ignoring {(~mask).sum()}/{len(mask)} fields with area < {min_area_to_ignore}m2")

        labels = labels.loc[mask]
        labels = labels.to_crs(crs)

        for index, feature in tqdm(labels.iterrows(), total=len(labels), position=0, leave=True, desc="INFO: Extracting time series into the folder: {}".format(npyfolder)):

            npyfile = os.path.join(npyfolder, "fid_{}.npz".format(feature.fid))
            if not os.path.exists(npyfile):

                left, bottom, right, top = feature.geometry.bounds
                window = rio.windows.from_bounds(left, bottom, right, top, transform)

                # reads each tif in tifs on the bounds of the feature. shape T x D x H x W
                image_stack = np.stack([rio.open(tif).read(window=window) for tif in tifs])

                with rio.open(tifs[0]) as src:
                    win_transform = src.window_transform(window)

                out_shape = image_stack[0, 0].shape
                assert out_shape[0] > 0 and out_shape[1] > 0, "WARNING: fid:{} image stack shape {} is zero in one dimension".format(feature.fid,image_stack.shape)

                # rasterize polygon to get positions of field within crop
                mask = features.rasterize(feature.geometry, all_touched=True,transform=win_transform, out_shape=image_stack[0, 0].shape)

                #mask[mask != feature.fid] = 0
                #mask[mask == feature.fid] = 1
                os.makedirs(npyfolder, exist_ok=True)
                np.savez(npyfile, image_stack=image_stack, mask=mask, feature=feature.drop("geometry").to_dict())


        return labels


if __name__ == '__main__':
    zippath = "/local_home/kuzu_ri/GIT_REPO/starter-pack-ai4food-v0.0.1/notebook/data/planet/UTM-24000/33N/18E-242N/PF-SR/"
    labelgeojson = "/local_home/kuzu_ri/GIT_REPO/starter-pack-ai4food-v0.0.1/notebook/data/brandenburg-gt/brandenburg_crops_train_2018.geojson"
    ds = PlanetReader(zippath, labelgeojson, selected_time_points=[2,3,4])
    len(ds)
    X,y,fid = ds[0]