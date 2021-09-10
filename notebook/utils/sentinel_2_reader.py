import os
import torch
from torch.utils.data import Dataset
import zipfile
from sh import gunzip
from glob import glob
import pickle
import sentinelhub # this import is necessary for pickle loading
import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio import features
from tqdm import tqdm


class S2Reader(Dataset):
    def __init__(self, zippath, labelgeojson,label_ids=None, transform=None):

        self.crop_ids = label_ids
        if label_ids is not None:
            self.crop_ids = label_ids.tolist()

        npzcache = zippath.replace(".zip", ".npz")

        self.data_transform = transform

        self.labels = gpd.read_file(labelgeojson)

        if not os.path.exists(npzcache):
            self.tsdata, self.clouddata, self.fids, self.crop_ids = S2Reader._etup(zippath, labelgeojson)
            print(f"saving extracted time series with label data to {npzcache}")
            np.savez(npzcache, tsdata=self.tsdata, clouddata=self.clouddata, fids=self.fids, crop_ids=self.crop_ids)
        else:
            self.tsdata = np.load(npzcache)["tsdata"]
            self.clouddata = np.load(npzcache)["clouddata"]
            self.fids = np.load(npzcache)["fids"]
            self.crop_ids = np.load(npzcache)["crop_ids"]

    def __len__(self):
        return len(self.fids)

    def __getitem__(self, item):
        X = self.tsdata[item]
        y = self.crop_ids[item]
        cld = self.clouddata[item]
        fid = self.fids[item]

        if self.data_transform is not None:
            X = self.data_transform(X, cld)

        if self.crop_ids is not None:
            label = self.crop_ids.index(y)
        else:
            label = y

        return X, label, fid

    @staticmethod
    def _setup(zippath, labelgeojson):
        """
        This utility function unzipps a dataset from Sinergize and performs a field-wise aggregation.
        results are written to a .npz cache with same name as zippath
        """
        datadir = os.path.dirname(zippath)
        rootpath = zippath.replace(".zip", "")
        if not (os.path.exists(rootpath) and os.path.isdir(rootpath)):
            print(f"unzipping {zippath} to {datadir}")
            with zipfile.ZipFile(zippath, 'r') as zip_ref:
                zip_ref.extractall(datadir)
        else:
            print(f"found folder in {rootpath}, no need to unzip")

        # find all .gz-ipped files and unzip
        for gz in glob(os.path.join(rootpath,"*","*.gz")) + glob(os.path.join(rootpath,"*.gz")):
            print(f"unzipping {gz}")
            gunzip(gz)

        with open(os.path.join(rootpath, "bbox.pkl"), 'rb') as f:
            bbox = pickle.load(f)
            crs = str(bbox.crs)
            minx, miny, maxx, maxy = bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y

        labels = gpd.read_file(labelgeojson)
        # project to same coordinate reference system (crs) as the imagery
        labels = labels.to_crs(crs)

        bands = np.load(os.path.join(rootpath, "data", "BANDS.npy"))
        clp = np.load(os.path.join(rootpath, "data", "CLP.npy"))
        #bands = np.concatenate([bands, clp], axis=-1) # concat cloud probability
        _, width, height, _ = bands.shape

        transform = rio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

        fid_mask = features.rasterize(zip(labels.geometry, labels.fid), all_touched=True,
                                  transform=transform, out_shape=(width, height))
        assert len(np.unique(fid_mask)) > 0, f"vectorized fid mask contains no fields. " \
                                             f"Does the label geojson {labelgeojson} cover the region defined by {zippath}?"

        crop_mask = features.rasterize(zip(labels.geometry, labels.crop_id), all_touched=True,
                                  transform=transform, out_shape=(width, height))
        assert len(np.unique(crop_mask)) > 0, f"vectorized fid mask contains no fields. " \
                                              f"Does the label geojson {labelgeojson} cover the region defined by {zippath}?"

        fids = []
        crop_ids = []
        tsdata = []
        clouddata = []
        for fid, crop_id in tqdm(zip(labels.fid.unique(), labels.crop_id.values), total=len(labels), desc="extracting time series"):
            field_mask = fid_mask == fid
            if field_mask.sum() > 0:
                data = bands.transpose(0, 3, 1, 2)[:, :, field_mask].mean(-1)
                tsdata.append(data)
                clouddata.append(clp.transpose(0,3,1,2)[:,:,field_mask].mean(-1))
                crop_ids.append(crop_id)
                fids.append(fid)
            else:
                print(f"field {fid} contained no pixels. Is it too small with {labels.loc[labels.fid==fid].geometry.area}m2 ? skipping...")

        tsdata = np.stack(tsdata)
        clouddata = np.stack(clouddata)
        return tsdata, clouddata, fids, crop_ids

if __name__ == '__main__':
    zippath = "s2-utm-33N-18E-242N-2018.zip"
    labelgeojson = "brandenburg_crops_train_2018.geojson"
    ds = S2Reader(zippath, labelgeojson)
    len(ds)
    X,y,fid = ds[0]
