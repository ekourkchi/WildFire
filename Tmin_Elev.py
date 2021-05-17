from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
from george import kernels
import george
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
import matplotlib
import sys
import os
import matplotlib.pyplot as plt
import requests
import json
import pandas as pd
from pandas.io import sql
from pandas.io.json import json_normalize
import numpy as np
from sqlalchemy import types, create_engine
from datetime import date, timedelta
from datetime import datetime
import time
import re
import pylab as py
from matplotlib import gridspec
import matplotlib.dates as md
from sklearn.model_selection import LeaveOneOut
os.environ['PROJ_LIB'] = '/home/ehsan/anaconda3/share/proj'


def linear(x, a, b):
    return a * x + b


def bilinear(x, a, b, c):

    left = a * x + b
    right = c * (x - 2150) + (a * 2150 + b)

    try:
        y = np.asarray([left[i] if x[i] <= 2150 else right[i]
                        for i in range(len(x))])
        return y
    except BaseException:
        if x <= 2150:
            return left
        else:
            return right


# calculate bic for regression
def calculate_bic(n, mse, num_params):
    bic = n * np.log(mse) + num_params * np.log(n)
    return bic

# calculate aic for regression


def calculate_aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    return aic


def metrics(y1, y2, verbose=True, n_param=1):
    '''
    y1 and y2 are two series of the same size

    This function outputs the MAE, RMSE and R^2
    of the cross evaluated series.

    '''
    y1 = y1.reshape(-1)
    y2 = y2.reshape(-1)

    n = len(y1)

    mse = np.mean((y1 - y2)**2)

    RMSE = np.sqrt(mse)
    MAE = np.mean(np.abs(y1 - y2))
    R2 = np.max([r2_score(y1, y2), r2_score(y2, y1)])

    BIC = calculate_bic(n, mse, n_param)
    AIC = calculate_aic(n, mse, n_param)

    if verbose:
        print('MAE: %.2f' % MAE, ' RMSE: %.2f' % RMSE, ' R^2: %.2f' % R2)
        print('AIC: %.2f' % AIC, ' BIC: %.2f' % BIC)

    return MAE, RMSE, R2, AIC, BIC


####################################################################

def nll_fn2(X, y):

    def step(theta):

        loo = KFold(n_splits=3)
        # loo = LeaveOneOut()
        XI2 = 0

        for train_index, test_index in loo.split(X):
        #if True: # for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            N = X_train.shape[0]

            zp = theta[0]
            slope = theta[1]
            s2 = theta[2]

            y_model = slope * X_train[:, 2] + zp
            ind, = np.where(X_train[:, 2] > 2150)
            y_model[ind] += s2 * (X_train[:, 2][ind] - 2150)

            y_model_test = slope * X_test[:, 2] + zp
            ind, = np.where(X_test[:, 2] > 2150)
            y_model_test[ind] += s2 * (X_test[:, 2][ind] - 2150)

            delta = np.abs(y_model_test - y_test)

#             print(delta[0])
#             if delta[0] > 3:
#                 delta[0]=0

            XI2 += np.sum(delta**2)

        return XI2

    return step

####################################################################


if __name__ == '__main__':

    iCODE = str(sys.argv[1])

    ISLAND_code = [iCODE]
    if iCODE == 'MA':
        ISLAND_code = ['MA', 'KO', 'MO', 'LA']

    pr = pd.read_csv("tmin_predictors.csv",
                     encoding="ISO-8859-1", engine='python')
    pr = pr.set_index("SKN")
    pr = pr[(pr.Island.isin(ISLAND_code))]

    rf = pd.read_csv("2_Partial_Fill_Daily_RF_mm_1990_2020.csv",
                     encoding="ISO-8859-1", engine='python')
    meta_columns = [
        'SKN',
        'Station.Name',
        'Observer',
        'Network',
        'Island',
        'ELEV.m.',
        'LAT',
        'LON',
        'NCEI.id',
        'NWS.id',
        'NESDIS.id',
        'SCAN.id',
        'SMART_NODE_RF.id']

    Temp_columns = rf.columns[13:]

    rf2 = rf[meta_columns]
    rf2 = rf2.set_index("SKN")

    rf2 = rf2[(rf2.Island.isin(ISLAND_code))]

    rf1 = rf[["SKN"] + list(Temp_columns)].T

    new_header = rf1.iloc[0]
    rf1 = rf1[1:]
    rf1.columns = new_header

    rf1.index = pd.to_datetime([x.split('X')[1] for x in rf1.index.values])
    rf1.index.name = 'Date'

    rf1 = rf1[list(rf2.index.values)]

    rf3 = rf2[["LON", "LAT", "ELEV.m."]].T
    rf3 = rf3[list(rf2.index.values)]

    rf_station = rf3.T
    rf_station = rf_station.join(rf1.T, how='left')
    rf_station.head()

    df = pd.read_csv("Tmin_QC.csv", encoding="ISO-8859-1", engine='python')

    meta_columns = [
        'SKN',
        'Station.Name',
        'Observer',
        'Network',
        'Island',
        'ELEV.m.',
        'LAT',
        'LON',
        'NCEI.id',
        'NWS.id',
        'NESDIS.id',
        'SCAN.id',
        'SMART_NODE_RF.id']
    Temp_columns = df.columns[13:]

    df2 = df[meta_columns]
    df2 = df2.set_index("SKN")

    df2 = df2[(df2.Island.isin(ISLAND_code))]

    df1 = df[["SKN"] + list(Temp_columns)].T

    new_header = df1.iloc[0]
    df1 = df1[1:]
    df1.columns = new_header

    df1.index = pd.to_datetime([x.split('X')[1] for x in df1.index.values])
    df1.index.name = 'Date'

    df1 = df1[list(df2.index.values)]

    df3 = df2[["LON", "LAT", "ELEV.m."]].T
    df3 = df3[list(df2.index.values)]

    df_station = df3.T
    df_station = df_station.join(df1.T, how='left')

    dt_cols = df_station.columns[3:][::-1]

    t1 = datetime.now()
    ###########################################
    Hyper = {}
    Hyper['MAE'] = []
    Hyper['RMSE'] = []
    Hyper['R2'] = []
    Hyper['AIC'] = []
    Hyper['BIC'] = []
    Hyper['t0'] = []
    Hyper['t1'] = []
    Hyper['t2'] = []
    ###########################################

    for dt_col in dt_cols:

        try:
            date_str = str(dt_col.date())

            Date = pd.to_datetime(date_str)

            data_rf = rf_station[[Date]].dropna()

            data_rf = data_rf.rename(columns={Date: 'RF'})

            data_df = df_station[["LON", "LAT", "ELEV.m.", Date]].dropna()

            data_df = data_df.rename(columns={Date: 'T'})

            df = data_df.join(
                data_rf,
                how='left',
                lsuffix='_T',
                rsuffix='_RF').dropna()

            df = df.join(pr[pr.columns[:-3]], how='left').dropna()

            X = df[["LON", "LAT", "ELEV.m."]].values
            y = df['T'].values

            if len(data_df) > 1:

                u = np.arange(np.round(np.max(X[:, 2])))

                fit, cov = curve_fit(linear, X[:, 2], y, sigma=y * 0 + 1)
                v1 = linear(u, fit[0], fit[1])

                fit, cov = curve_fit(bilinear, X[:, 2], y, sigma=y * 0 + 1)

                model = bilinear(X[:, 2], fit[0], fit[1], fit[2])

                indx, = np.where(np.abs(model - y) < 3)

                fit, cov = curve_fit(
                    bilinear, X[:, 2][indx], y[indx], sigma=y[indx] * 0 + 1)

                model = bilinear(X[:, 2], fit[0], fit[1], fit[2])

                indx, = np.where(np.abs(model - y) < 3)

                fit, cov = curve_fit(
                    bilinear, X[:, 2][indx], y[indx], sigma=y[indx] * 0 + 1)

                v2 = bilinear(u, fit[0], fit[1], fit[2])

                X = X[indx]
                y = y[indx]

                # Maximum Likelihood
                pos = minimize(nll_fn2(X, y), [
                               0, -0.002, 0], method='L-BFGS-B')

                theta = pos.x

                N = X.shape[0]
                zp = theta[0]
                slope = theta[1]
                s2 = theta[2]

                y_model = slope * X[:, 2] + zp
                ind, = np.where(X[:, 2] > 2150)
                y_model[ind] += s2 * (X[:, 2][ind] - 2150)

                MAE, RMSE, R2, AIC, BIC = metrics(
                    y, y_model, verbose=False, n_param=len(theta))

                if pos.success:
                    Hyper['MAE'].append(MAE)
                    Hyper['RMSE'].append(RMSE)
                    Hyper['R2'].append(R2)
                    Hyper['AIC'].append(AIC)
                    Hyper['BIC'].append(BIC)
                    Hyper['t0'].append(theta[0])
                    Hyper['t1'].append(theta[1])
                    Hyper['t2'].append(theta[2])

                    pd.DataFrame.from_dict(Hyper).to_csv(
                        'Tmin_Elev0_'+iCODE+'_hyper.csv', sep=',', index=False)
                # print(date_str, pos.success)

        except:
            pass
