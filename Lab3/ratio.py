import numpy as np
from osgeo import gdal

b = gdal.Open("bosnia-ndbi.tif").ReadAsArray()
a = gdal.Open("bosnia-ndvi.tif").ReadAsArray(buf_xsize=b.shape[1], buf_ysize=b.shape[0], resample_alg=gdal.GRIORA_Average)
ok = (abs(a) <= 1) & (abs(b) <= 1)
print("Mittel NDBI - NDVI:", (b - a)[ok].mean())
