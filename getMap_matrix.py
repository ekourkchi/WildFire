import rasterio
from affine import Affine
from pyproj import Proj, transform
from osgeo import gdal
import numpy as np



def getmap(fname):

	# Read raster
	with rasterio.open(fname) as r:
	    T0 = r.transform  # upper-left pixel corner affine transform
	    p1 = Proj(r.crs)
	    A = r.read()  # pixel values

	# All rows and columns
	cols, rows = np.meshgrid(np.arange(A.shape[2]), np.arange(A.shape[1]))

	# Get affine transform for pixel centres
	T1 = T0 * Affine.translation(0.5, 0.5)
	# Function to convert pixel row/column index (from 0) to easting/northing at centre
	rc2en = lambda r, c: (c, r) * T1

	# All eastings and northings (there is probably a faster way to do this)
	eastings, northings = np.vectorize(rc2en, otypes=[np.float, np.float])(rows, cols)

	# Project all longitudes, latitudes
	p2 = Proj(proj='latlong', datum='WGS84')
	longs, lats = transform(p1, p2, eastings, northings)	

	ds = gdal.Open(fname)
	myarray = np.array(ds.GetRasterBand(1).ReadAsArray())

	m, n = myarray.shape

	longs_1d = longs.reshape(-1)
	lats_1d  = lats.reshape(-1)
	eleve_1d = myarray.reshape(-1)

	N = len(longs_1d)

	X_grid = np.zeros((N, 3))
	X_grid[:,0] = longs_1d
	X_grid[:,1] = lats_1d
	X_grid[:,2] = eleve_1d

	return X_grid













