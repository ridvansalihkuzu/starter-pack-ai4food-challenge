import os
from torch.utils.data import Dataset
import zipfile
from sh import gunzip
from glob import glob
import pickle
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features
from tqdm import tqdm


class S1Reader(Dataset):
    def __init__(self, input_dir, label_dir, label_ids=None, transform=None, min_area_to_ignore = 1000, selected_time_points=None):

        self.data_transform = transform
        self.selected_time_points=selected_time_points
        self.crop_ids = label_ids
        if label_ids is not None:
            self.crop_ids = label_ids.tolist()

        self.npyfolder = input_dir.replace(".zip", "/time_series")
        self.labels = S1Reader._setup(input_dir, label_dir,self.npyfolder,min_area_to_ignore)

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
    def _setup(zippath, labelgeojson, npyfolder, min_area_to_ignore=1000):
        """
        This utility function unzipps a dataset from Sinergize and performs a field-wise aggregation.
        results are written to a .npz cache with same name as zippath
        """
        datadir = os.path.dirname(zippath)
        rootpath = zippath.replace(".zip", "")
        if not (os.path.exists(rootpath) and os.path.isdir(rootpath)):
            print(f"INFO: Unzipping {zippath} to {datadir}")
            with zipfile.ZipFile(zippath, 'r') as zip_ref:
                zip_ref.extractall(datadir)
        else:
            print(f"INFO: Found folder in {rootpath}, no need to unzip")

        # find all .gz-ipped files and unzip
        for gz in glob(os.path.join(rootpath,"*","*.gz")) + glob(os.path.join(rootpath,"*.gz")):
            print(f"INFO: Unzipping {gz}")
            gunzip(gz)

        with open(os.path.join(rootpath, "bbox.pkl"), 'rb') as f:
            bbox = pickle.load(f)
            crs = str(bbox.crs)
            minx, miny, maxx, maxy = bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y

        labels = gpd.read_file(labelgeojson)
        # project to same coordinate reference system (crs) as the imagery
        ignore = labels.geometry.area > min_area_to_ignore
        print(f"INFO: Ignoring {(~ignore).sum()}/{len(ignore)} fields with area < {min_area_to_ignore}m2")
        labels = labels.loc[ignore]
        labels = labels.to_crs(crs)

        vv = np.load(os.path.join(rootpath, "data", "VV.npy"))
        vh = np.load(os.path.join(rootpath, "data", "VH.npy"))
        bands = np.stack([vv[:,:,:,0],vh[:,:,:,0]], axis=3)
        _, width, height, _ = bands.shape

        bands=bands.transpose(0, 3, 1, 2)

        transform = rio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

        fid_mask = features.rasterize(zip(labels.geometry, labels.fid), all_touched=True,
                                  transform=transform, out_shape=(width, height))
        assert len(np.unique(fid_mask)) > 0, f"WARNING: Vectorized fid mask contains no fields. " \
                                             f"Does the label geojson {labelgeojson} cover the region defined by {zippath}?"

        crop_mask = features.rasterize(zip(labels.geometry, labels.crop_id), all_touched=True,
                                  transform=transform, out_shape=(width, height))
        assert len(np.unique(crop_mask)) > 0, f"WARNING: Vectorized fid mask contains no fields. " \
                                              f"Does the label geojson {labelgeojson} cover the region defined by {zippath}?"


        for index, feature in tqdm(labels.iterrows(), total=len(labels), position=0, leave=True, desc="INFO: Extracting time series into the folder: {}".format(npyfolder)):

            npyfile = os.path.join(npyfolder, "fid_{}.npz".format(feature.fid))
            if not os.path.exists(npyfile):

                left, bottom, right, top = feature.geometry.bounds
                window = rio.windows.from_bounds(left, bottom, right, top, transform)

                row_start = round(window.row_off)
                row_end = round(window.row_off) + round(window.height)
                col_start = round(window.col_off)
                col_end = round(window.col_off) + round(window.width)

                image_stack = bands[:, :,row_start:row_end, col_start:col_end]
                mask = fid_mask[row_start:row_end, col_start:col_end]
                mask[mask != feature.fid] = 0
                mask[mask == feature.fid] = 1
                os.makedirs(npyfolder, exist_ok=True)
                np.savez(npyfile, image_stack=image_stack, mask=mask, feature=feature.drop("geometry").to_dict())

        return labels




if __name__ == '__main__':
    zippath = "/local_home/kuzu_ri/GIT_REPO/starter-pack-ai4food-v0.0.1/notebook/data/sentinel-1/s1-asc-utm-33N-18E-242N-2018.zip"
    labelgeojson = "/local_home/kuzu_ri/GIT_REPO/starter-pack-ai4food-v0.0.1/notebook/data/brandenburg-gt/brandenburg_crops_train_2018.geojson"
    ds = S1Reader(zippath, labelgeojson,selected_time_points=[1,2,3])
    len(ds)
    X,y,fid = ds[1]
