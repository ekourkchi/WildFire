#!/usr/bin/env python
# coding: utf-8

# In[1]:


import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
from osgeo import gdal
import pandas as pd
from affine import Affine
from pyproj import Proj, transform, Transformer
import os
from astropy.table import Table, Column

from Temp_linear import *

matplotlib.use('Agg')   # changing the matplotlib backend
# In[2]:


# Set Axes
def set_axes(ax, xlim=None, ylim=None, fontsize=16,
             twinx=True, twiny=True, minor=True, inout='in'):

    if not ylim is None:
        ax.set_ylim(ylim)
    else:
        ylim = ax.get_ylim()

    if not xlim is None:
        ax.set_xlim(xlim)
    else:
        xlim = ax.get_xlim()

    ax.tick_params(which='major', length=6, width=1., direction=inout)
#         if minor:
    ax.tick_params(
        which='minor',
        length=3,
        color='#000033',
        width=1.0,
        direction=inout)

    if twiny:
        y_ax = ax.twinx()
        y_ax.set_ylim(ylim)
        y_ax.set_yticklabels([])
        y_ax.minorticks_on()
        y_ax.tick_params(which='major', length=6, width=1., direction=inout)
        if minor:
            y_ax.tick_params(
                which='minor',
                length=3,
                color='#000033',
                width=1.0,
                direction=inout)

    if twinx:
        x_ax = ax.twiny()
        x_ax.set_xlim(xlim)
        x_ax.set_xticklabels([])
        x_ax.minorticks_on()
        x_ax.tick_params(which='major', length=6, width=1.0, direction=inout)
        if minor:
            x_ax.tick_params(
                which='minor',
                length=3,
                color='#000033',
                width=1.0,
                direction=inout)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    return x_ax, y_ax


# In[3]:


def get_Coordinates(GeoTiff_name):

    # Read raster
    with rasterio.open(GeoTiff_name) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
        A = r.read()  # pixel values

    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[2]), np.arange(A.shape[1]))

    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing
    # at centre
    def rc2en(r, c): return T1 * (c, r)

    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(
        rc2en, otypes=[
            np.float, np.float])(
        rows, cols)

    transformer = Transformer.from_proj(
        'EPSG:4326',
        '+proj=longlat +datum=WGS84 +no_defs +type=crs',
        always_xy=True,
        skip_equivalent=True)

    LON, LAT = transformer.transform(eastings, northings)
    return LON, LAT


# In[4]:


def genParamCode(param):

    if param == 'coastDist':
        return 'coastDistM'
    elif param == 'rf':
        return 'meanAnn'
    else:
        return param


# In[5]:


def genTiffName(param='dem', iCode='bi'):

    if param == 'dem_250':
        param = 'dem'

    pCODE = genParamCode(param)
    TiffName = r'./geoTiffs_250m/' + param + \
        '/' + iCode + '_' + pCODE + '_250m.tif'

    return TiffName


# In[6]:


def getDataArray(param='dem_250', iCode='bi', getMask=False):

    TiffName = genTiffName(param=param, iCode=iCode)
    raster_img = rasterio.open(TiffName)

    myarray = raster_img.read(1)
    msk = raster_img.read_masks(1)

    msk[msk > 0] = 1
    dataArray = myarray * msk

    dataArray[msk == 0] = 0

    if getMask:
        return msk, myarray.shape    # 0:reject  >0:accept

    return dataArray


# In[7]:


def get_island_grid(iCode, params):

    TiffName = genTiffName(iCode=iCode)
    LON, LAT = get_Coordinates(TiffName)
    LON = LON.reshape(-1)
    LAT = LAT.reshape(-1)

    myDict = {'LON': LON, 'LAT': LAT}

    for param in params:

        myDict[param] = getDataArray(iCode=iCode, param=param).reshape(-1)

    island_df = pd.DataFrame.from_dict(myDict)

    mask, shape = getDataArray(iCode=iCode, getMask=True)

    return island_df, mask.reshape(-1), shape


# In[8]:


def G_islandName(iCode):

    if iCode == 'bi':
        return "Big Island"
    elif iCode == 'oa':
        return "Oahu"
    elif iCode == 'mn':
        return "Maui+"
    elif iCode == 'ka':
        return "Kauai"
    else:
        return iCode


def generate_outputs(iCode, mode, params, date_str, outputFolder):

    # iCode = 'bi'
    # mode = 'max'
    # params = ["dem_250"] # , "rf", "lai", "tpi", "rf", "albedo"]
    # date_str = '2005-08-10'

    dateSpl = date_str.split('-')
    dateTail = dateSpl[0][2:] + dateSpl[1] + dateSpl[2]

    Tmap_tiff = './' + outputFolder + '/T' + mode + \
        '_map_' + iCode + '_' + dateTail + '.tiff'
    Tmap_png = './' + outputFolder + '/T' + mode + \
        '_map_' + iCode + '_' + dateTail + '.png'
    Tmap_fig1 = './' + outputFolder + '/T' + mode + \
        '_fig1_' + iCode + '_' + dateTail + '.png'
    Tmap_fig2 = './' + outputFolder + '/T' + mode + \
        '_fig2_' + iCode + '_' + dateTail + '.png'
    Tmap_log = './' + outputFolder + '/T' + \
        mode + '_' + iCode + '_' + dateTail + '.log'

    threshold = 2.5     # threshold of sigma clipping when removing outliers
    inversion = 2150    # meter
    mixHighAlt = 2150   # meter

    predictors = "t" + mode + "_predictors.csv"
    temperature = "T" + mode + "_QC.csv"
    rain = "2_Partial_Fill_Daily_RF_mm_1990_2020.csv"
    inputs = [predictors, temperature, rain]

    island_df, mask, shape = get_island_grid(iCode, params)

    param_List = ["Island", "LON", "LAT"] + params

    df_station, rf_station, pr_station, ISLAND_code = Data_prep(iCode.upper(),
                                                                mode,
                                                                inputFileList=inputs,
                                                                mixHighAlt=mixHighAlt)
    # df_station, rf_station, pr_station, ISLAND_code = Data_prep(['MA', 'KO', 'MO', 'LA', 'BI'], mode, codeIsList=True)

    df_date = prep_Data(date_str, df_station, rf_station, pr_station)

    MODEL = myModel(inversion=inversion)
    theta, cov, X, y = makeModel(
        df_date, param_List, MODEL, threshold=threshold)

    # In[10]:

    # In[12]:

    fig = pyplot.figure(figsize=(7, 4), dpi=80)
    fig.subplots_adjust(wspace=0, top=0.9, bottom=0.12, left=0.1, right=0.7)
    ax = fig.add_subplot(111)

    N = int(np.round(np.max(X[:, 3])))
    _, n_params = X.shape
    u = np.zeros((N, n_params))
    u[:, 3] = np.arange(N)
    v = MODEL(u[:, 3:], *theta)

    ax.plot(u[:, 3], v, 'k:')

    df_date = df_date[["T"] + param_List].dropna()
    X = df_date[param_List].values
    y = df_date['T'].values
    ax.plot(X[:, 3], y, 'ko', mfc='white', ms=8)

    df_BI = df_date[~df_date.Island.isin(ISLAND_code)]
    X0 = df_BI[param_List].values
    y0 = df_BI['T'].values
    ax.plot(X0[:, 3], y0, 'ro', mfc='white', ms=8)

    # get the indices of acceptable data points
    indx = removeOutlier(X, y, threshold=threshold)
    X1 = X[indx]
    y1 = y[indx]
    ax.plot(X1[:, 3], y1, 'k.', mfc='k')

    N = int(np.round(np.max(X[:, 3])))
    u = np.zeros((N, n_params))
    u[:, 3] = np.arange(N)
    v = MODEL(u[:, 3:], *theta)
    ax.plot(u[:, 3], v, label='Bilinear fit')

    ax.set_xlabel("Elevation (m)", fontsize=14)
    ax.set_ylabel(r"$T_{" + mode + r"} \/\/[^oC]$", fontsize=14)
    ax.set_title(G_islandName(iCode) + "  [" + date_str + "]", fontsize=14)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.plot([-1000], [-1000], 'ko', mfc='white',
            ms=8, label=G_islandName(iCode))
    ax.plot([-1000], [-1000], 'ro', mfc='white', ms=8, label='Other Islands')
    ax.plot([-1000], [-1000], 'k.', mfc='k', label='Accepted')

    # Ylm = ax.get_ylim() ; Xlm = ax.get_xlim()
    yl2 = 0.15 * ylim[0] + 0.85 * ylim[1]
    yl1 = 0. * ylim[0] + 1 * ylim[1]
    ax.plot([inversion, inversion], [yl1, yl2], ':',
            color='maroon', label='Break point')
    yl2 = 0.85 * ylim[0] + 0.15 * ylim[1]
    yl1 = 1. * ylim[0] + 0 * ylim[1]
    ax.plot([inversion, inversion], [yl1, yl2], ':', color='maroon')

    set_axes(ax, xlim=xlim, ylim=ylim, fontsize=12)

    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=12)

    fig.savefig(Tmap_fig1, dpi=200)
    plt.close("all")

    # In[13]:

    fig = pyplot.figure(figsize=(5, 4), dpi=80)
    fig.subplots_adjust(wspace=0, top=0.9, bottom=0.12, left=0.15, right=0.95)
    ax = fig.add_subplot(111)

    ax.plot(y, MODEL(X[:, 3:], *theta), 'ko',
            mfc='white', ms=8, label=G_islandName(iCode))
    ax.plot(y0, MODEL(X0[:, 3:], *theta), 'ro',
            mfc='white', ms=8, label='Other Islands')
    ax.plot(y1, MODEL(X1[:, 3:], *theta), 'k.', mfc='k', label='Accepted')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    lim1 = np.min([xlim[0], ylim[0]]) - 1
    lim2 = np.max([xlim[1], ylim[1]]) + 1

    ax.plot([-10, 35], [-10, 35], 'k:', label='Equality')

    ax.set_xlabel(r"$Observed \/\/ T_{" + mode + r"} \/\/[^oC]$", fontsize=14)
    ax.set_ylabel(r"$Model \/\/ T_{" + mode + r"} \/\/[^oC]$", fontsize=14)
    ax.set_title(G_islandName(iCode) + " (" + date_str + ")", fontsize=14)

    set_axes(ax, xlim=(lim1, lim2), ylim=(lim1, lim2), fontsize=12)
    ax.legend(loc='upper left', fontsize=12)

    fig.savefig(Tmap_fig2, dpi=200)
    plt.close("all")

    # In[14]:

    df_date['flag'] = False
    temp = df_date.flag.values
    temp[indx] = True
    df_date['flag'] = temp
    df_date['T_model'] = MODEL(X[:, 3:], *theta)

    df_output = df_date[["T"] + param_List + ["flag", "T_model"]]
    df_output_ = df_output.reset_index()
    df_output_.to_csv(Tmap_log, sep=',', index=False)

    table = np.genfromtxt(Tmap_log, delimiter=',',
                          filling_values=-1000000, names=True, dtype=None, encoding=None)

    colnames = table.dtype.names

    # table is a structured array
    myTable = {}
    for name in table.dtype.names:
        myTable[name] = table[name]
    table = myTable
    # table is now a dictionary

    myTable = Table()

    for key in colnames:
        if key in ["SKN", "Island", "flag"]:
            myTable.add_column(Column(data=table[key], name=key))
        elif key in ['LON', "LAT"]:
            myTable.add_column(
                Column(
                    data=table[key],
                    name=key,
                    format='%0.4f'))
        else:
            myTable.add_column(
                Column(
                    data=table[key],
                    name=key,
                    format='%0.3f'))

    # to be used on EDD
    myTable.write(
        Tmap_log,
        format='ascii.fixed_width',
        delimiter='|',
        bookend=False,
        overwrite=True)

    with open(Tmap_log, "r") as File:
        reader = File.read()

    # In[15]:

    n_data, n_param = X1[:, 3:].shape
    MAE, RMSE, R2, AIC, AICc, BIC = metrics(
        y1, MODEL(X1[:, 3:], *theta), n_param=n_param, n_data=n_data, verbose=False)

    # In[16]:

    with open(Tmap_log, 'w') as File:

        cols = df_output.columns

        s = " Equation: T = B0 + B1*" + \
            cols[4] + " + B2*" + cols[4] + "[z>" + str(inversion) + "]"

        for t in range(1, n_params - 2):
            s += " + B" + str(t + 2) + "*" + cols[t + 4]

        File.write(' ----------------------------------------------------')
        File.write('\n' + s)

        File.write("\n\n B: " + str(theta))
        File.write("\n Covariance:\n " + str(cov))

        File.write(
            '\n\n MAE: %.2f' %
            MAE +
            '\n RMSE: %.2f' %
            RMSE +
            '\n R^2: %.2f' %
            R2)
        File.write(
            '\n AIC: %.2f' %
            AIC +
            '\n AICc: %.2f' %
            AICc +
            '\n BIC: %.2f' %
            BIC)

        File.write('\n ----------------------------------------------------')

        File.write("\n\n")

        File.write(reader)

    # In[ ]:

    # In[26]:

    df_date['flag'] = False
    temp = df_date.flag.values
    temp[indx] = True
    df_date['flag'] = temp

    # In[30]:

    X_island = island_df.values

    T_model = MODEL(X_island[:, 2:], *theta)

    # In[31]:

    T_model[mask == 0] = np.nan

    # In[32]:

    cols, rows = shape

    # In[33]:

    fp = genTiffName(param='dem', iCode=iCode)

    ds = gdal.Open(fp)
    cols, rows = shape

    # arr_out = np.where((arr < arr_mean), -100000, arr)
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(Tmap_tiff, rows, cols, 1, gdal.GDT_Float64)
    # sets same geotransform as input
    outdata.SetGeoTransform(ds.GetGeoTransform())
    outdata.SetProjection(ds.GetProjection())  # sets same projection as input
    outdata.GetRasterBand(1).WriteArray(T_model.reshape(shape))
    # if you want these values (in the mask) transparent
    outdata.GetRasterBand(1).SetNoDataValue(0)
    outdata.FlushCache()  # saves to disk!!
    outdata = None
    band = None
    ds = None

    # In[34]:

    fig = pyplot.figure(figsize=(9, 9), dpi=80)
    ax = fig.add_subplot(1, 1, 1)

    fp = Tmap_tiff

    raster_img = rasterio.open(fp)

    myarray = raster_img.read(1)
    msk = raster_img.read_masks(1)

    msk[msk > 0] = 1
    image = myarray * msk

    img = ax.imshow(image, cmap='viridis')
    show(raster_img, ax=ax, cmap='viridis')

    cbar = fig.colorbar(img, ax=ax, shrink=0.6)

    cbar.set_label(
        r'$Temperature \/\/ [^oC]$',
        rotation=90,
        fontsize=14,
        labelpad=10)
    cbar.ax.tick_params(labelsize=12)

    # set_axes(ax, (21.2, 21.7), (-158.3, -157.))

    ax.set_xlabel("Longitude [deg]", fontsize=14)
    ax.set_ylabel("Latitude [deg]", fontsize=14)
    # ax.set_title("Big Island (LAI - 250 m)", fontsize=16)

    fontsize = 13

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    ax.set_title(G_islandName(iCode) + " (" + date_str + ")", fontsize=14)

    fig.savefig(Tmap_png, dpi=200)
    plt.close("all")


# This code has been automatically covnerted to comply with the pep8 convention
# This the Linux command:
# $ autopep8 --in-place --aggressive  <filename>.py
if __name__ == '__main__':

    iCode = str(sys.argv[1])  # 'bi'
    mode = str(sys.argv[2])  # 'max'

    params = ["dem_250"]  # , "rf", "lai", "tpi", "rf", "albedo"]
    # date_str = '2005-08-10'

    outputFolder = iCode+'/'+mode+'/'
    iCODE = iCode.upper()

    df_station, rf_station, pr_station, ISLAND_code = Data_prep(iCODE, mode)
    dt_cols = df_station.columns[3:][::-1]

    for dt_col in dt_cols:
        try:
            date_str = str(dt_col.date())
            print(iCode, mode, date_str)
            generate_outputs(iCode, mode, params, date_str, outputFolder)
            print('Done -------------^ ')
        except BaseException:
            print('Error -------------^ ')
            pass
