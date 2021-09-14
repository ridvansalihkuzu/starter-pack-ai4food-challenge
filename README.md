# AI4Food - A Challenge for Crop Type Classification

## Get started with the challenge

This notebook will get you started with downloading, exploring and analysing the input and output data of the challenge.

The proposed challenge will focus on crop type classification based on a time-series input of _Sentinel-1_, _Sentinel-2_ and _Planet Fusion_ data. The challenge will cover two areas of interest, in Germany and South Africa, with high-quality cadastral data on field boundaries and crop types as ground truth input. 

The challenge will consist of two tracks:
  * Within-season crop identification, over the South Africa AOI
  * Reusability of models for crop identification from one growing season to the next, over the Germany AOI

The participants will not be required to participate in both challenges. However, the evaluation mechanism behind both tracks are the same, as well as the rules and prize catalogue.

This notebook showcases how to download and process the data, but you are free to use any open souce Python library specifically designed to deal with Earth Observation data such as [eo-learn](https://eo-learn.readthedocs.io/en/latest/index.html). In this notebook, the data is stored as `tif` images, `numpy` arrays and `geopandas` dataframes to facilitate processing operations and `torch` is preferred for data processing and training. However, you can use any other Python tool of preference to process the provided data.

The notebook also showcases how to generate a valid submission file.

As per challenge rules, the following applies:
 * no data source other than the ones provided can be used to produce your outputs;
 * pre-trained models are allowed;
 * the test data cannot be used for training of the models; 
 * the target _cultivated land_ map cannot be used as an input to the method (i.e. trivial solution).

Code for the winning solutions will be reviewed to ensure rules have been followed.

0. [Requirements](#requirements)


1. [Data overview](#data-overview)

   1.1. [Area of Interest for Brandenburg](#aoi1)
   
   1.2. [Data Types for Brandenburg](#dt1)
   
   1.3. [Area of Interest for South Africa](#aoi2)

   1.4. [Data Types for South Africa](#dt2)
   
   
2. [Data processing and ML Training](#data-processing)
    
   2.1. [Exploiting _Planet Fusion_ Data](#epfd)
   
   2.2. [Exploiting _Sentinel-2_ Data](#es2d)


3. [Prepare a Submission](#submission-example)

   3.1 [Submission Example](#prepare-a-submission)
