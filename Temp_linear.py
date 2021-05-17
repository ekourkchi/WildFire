import sys
import os
import time
import pylab as py
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, minimize
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
from datetime import date, timedelta
from datetime import datetime
from sklearn.model_selection import LeaveOneOut, KFold


# In[ ]:


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


# In[ ]:


# n: number of observations, number of data points
# num_param: number of free parameters in the model

# calculate bic for regression
def calculate_bic(n, mse, num_params):
    BIC = n * np.log(mse) + num_params * np.log(n)
    return BIC

# calculate aic for regression


def calculate_aic(n, mse, num_params):

    # for the linear regression, assuming that errors are normally distributed
    AIC = n * np.log(mse) + 2 * num_params

    AICc = AIC + 2 * num_params * (num_params + 1.) / (n - num_params - 1.)

    return AIC, AICc


def metrics(y1, y2, verbose=True, n_param=1, n_data=None):
    '''
    y1 and y2 are two series of the same size

    This function outputs the MAE, RMSE and R^2
    of the cross evaluated series.

    '''
    y1 = y1.reshape(-1)
    y2 = y2.reshape(-1)

    if n_data is None:
        n_data = len(y1)

    mse = np.mean((y1 - y2)**2)

    RMSE = np.sqrt(mse)
    MAE = np.mean(np.abs(y1 - y2))
    R2 = np.max([r2_score(y1, y2), r2_score(y2, y1)])

    BIC = calculate_bic(n_data, mse, n_param)
    AIC, AICc = calculate_aic(n_data, mse, n_param)

    if verbose:
        print('MAE: %.2f' % MAE, ' RMSE: %.2f' % RMSE, ' R^2: %.2f' % R2)
        print('AIC: %.2f' % AIC, 'AIC: %.2f' % AICc, ' BIC: %.2f' % BIC)

    return MAE, RMSE, R2, AIC, AICc, BIC


# In[ ]:


def removeOutlier(X, y, threshold=2.5):
    '''
     Fitting a lineat model based on elevation
     and clip outliers based on the given threshold
    '''

    fit, cov = curve_fit(bilinear, X[:, 3], y, sigma=y * 0 + 1)
    model = bilinear(X[:, 3], fit[0], fit[1], fit[2])
    stdev = np.std(model - y)  # 1-sigma scatter of residuals
    indx, = np.where(np.abs(model - y) < threshold * stdev)

    # repeating the process one more time to clip outliers based
    # on a more robust model
    fit, cov = curve_fit(
        bilinear, X[:, 3][indx], y[indx], sigma=y[indx] * 0 + 1)
    model = bilinear(X[:, 3], fit[0], fit[1], fit[2])
    stdev = np.std(model - y)
    indx, = np.where(np.abs(model - y) < threshold * stdev)

    return indx


# In[ ]:


def get_Temperature(temp_file, meta_columns, ISLAND_code, mixHighAlt=None):

    df = pd.read_csv(temp_file, encoding="ISO-8859-1", engine='python')

    Temp_columns = df.columns[13:]

    df2 = df[meta_columns]
    df2 = df2.set_index("SKN")

    # if 'BI' in ISLAND_code and not 'MA' in ISLAND_code:
    #     ISLAND_code += ['MA']
    # if 'MA' in ISLAND_code and not 'BI' in ISLAND_code:
    #     ISLAND_code += ['BI']

    if mixHighAlt is None:
        df2 = df2[(df2.Island.isin(ISLAND_code))]
    else:
        df2 = df2[((df2.Island.isin(ISLAND_code)) |
                   (df2["ELEV.m."] > mixHighAlt))]

    df1 = df[["SKN"] + list(Temp_columns)].T

    new_header = df1.iloc[0]
    df1 = df1[1:]
    df1.columns = new_header

    df1.index = pd.to_datetime([x.split('X')[1] for x in df1.index.values])
    df1.index.name = 'Date'

    df1 = df1[list(df2.index.values)]

    df3 = df2[["Island", "LON", "LAT", "ELEV.m."]].T
    df3 = df3[list(df2.index.values)]

    df_station = df3.T
    df_station = df_station.join(df1.T, how='left')

    return df_station


# In[ ]:


def get_Rain(
        rain_file,
        meta_columns,
        ISLAND_code,
        window=None,
        min_periods=None,
        mixHighAlt=None):

    rf = pd.read_csv(rain_file, encoding="ISO-8859-1", engine='python')

    Temp_columns = rf.columns[13:]

    rf2 = rf[meta_columns]
    rf2 = rf2.set_index("SKN")

    if mixHighAlt is None:
        rf2 = rf2[(rf2.Island.isin(ISLAND_code))]
    else:
        rf2 = rf2[((rf2.Island.isin(ISLAND_code)) |
                   (rf2["ELEV.m."] > mixHighAlt))]

    rf1 = rf[["SKN"] + list(Temp_columns)].T

    new_header = rf1.iloc[0]
    rf1 = rf1[1:]
    rf1.columns = new_header

    rf1.index = pd.to_datetime([x.split('X')[1] for x in rf1.index.values])
    rf1.index.name = 'Date'

    rf1 = rf1[list(rf2.index.values)]

    if window is not None:
        if min_periods is None:
            min_periods = 2 * window / 3
        elif min_periods > window:
            min_periods = window
        rf1 = rf1.rolling(str(window) + 'd', min_periods=min_periods).mean()

    rf3 = rf2[["Island", "LON", "LAT", "ELEV.m."]].T
    rf3 = rf3[list(rf2.index.values)]

    rf_station = rf3.T
    rf_station = rf_station.join(rf1.T, how='left')

    return rf_station


# In[ ]:


def get_Predictors(pred_file, ISLAND_code):

    pr = pd.read_csv(pred_file, encoding="ISO-8859-1", engine='python')
    pr = pr.set_index("SKN")
    # pr = pr[(pr.Island.isin(ISLAND_code))]

    return pr


# In[ ]:


def prep_Data(date_str, df_station, rf_station, pr):

    Date = pd.to_datetime(date_str)

    data_rf = rf_station[[Date]].rename(columns={Date: 'RF'})
    data_df = df_station[["LON", "LAT", "ELEV.m.", Date]].rename(columns={
        Date: 'T'})

    df = data_df.join(
        data_rf,
        how='left',
        lsuffix='_T',
        rsuffix='_RF')

    df = df.join(pr[pr.columns[:-3]], how='left')

    return df


# In[ ]:

def myModel(inversion=2150):
    '''
    This wrapper function constructs another function called "MODEL"
    according to the provided inversion elevation
    '''

    def MODEL(X, *theta):

        _, n_params = X.shape

        u = 2
        if inversion is None:
            u = 1

        y = theta[0] + theta[1] * X[:, 0]
        for t in range(1, n_params):
            y += theta[t+u] * X[:, t]

        if not inversion is None:
            ind, = np.where(X[:, 0] > inversion)
            y[ind] += theta[2] * (X[:, 0][ind] - inversion)

        return y

    return MODEL


# In[ ]:


def makeModel(df, param_List, MODEL, threshold=2.5, inversion=2150.):

    if True:  # try:

        # removing rows (stations) that do not have the model required
        # parameters
        df = df[["T"] + param_List].dropna()

        X = df[param_List].values
        y = df['T'].values

        n_data, n_params = X.shape

        if inversion is None:
            n_params -= 1


        if len(y) > 1:

            # get the indices of acceptable data points
            indx = removeOutlier(X, y, threshold=threshold)
            X = X[indx]
            y = y[indx]

            fit, cov = curve_fit(
                MODEL, X[:, 3:], y, p0=[30, -0.002] + (n_params - 3) * [0])

            return fit, cov, X, y

#     except:
#         pass

    return None, None, None, None


# In[ ]:

# calcualte metrics based on a leave one out strategy
def loo_Metrics(X, y, MODEL):

    loo = LeaveOneOut()
    n_data, n_params = X.shape

    u = np.zeros(n_data)
    v = np.zeros(n_data)
    i = 0

    for train_index, test_index in loo.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        fit, cov = curve_fit(MODEL, X_train[:, 3:], y_train, p0=[
                             30, -0.002] + (n_params - 3) * [0])

        y_loo = MODEL(X_test[:, 3:], *fit)

        u[i] = y_loo[0]
        v[i] = y_test[0]

        i += 1

    u, v = sigma_Clip(u, v)

    return metrics(u, v, verbose=False, n_param=len(fit), n_data=len(y))


# In[ ]:

def sigma_Clip(u, v, threshold=3.0):

    # removing 10% upper and lower quantiles of residuals (removing aggressive
    # outliers)
    delta = u - v
    indx = np.argsort(delta)
    u = u[indx]
    v = v[indx]
    N = len(u)
    i = int(np.ceil(1 * N / 10))
    j = int(np.floor(9 * N / 10))
    u = u[i:j]
    v = v[i:j]

    # Here we do some sigma clipping (assuming that residuals are normally
    # distributed)
    delta = u - v
    mean = np.median(delta)
    std = np.std(delta)
    indx = (
        (delta > mean -
         threshold *
         std) & (
            delta < mean +
            threshold *
            std))
    u = u[indx]
    v = v[indx]

    return u, v


# In[ ]:

def Data_prep(iCODE, mode, codeIsList=False,
              mixHighAlt=None, inputFileList=None):

    if inputFileList is None:
        pred_file = "t" + mode + "_predictors.csv"
        temperature_file = "T" + mode + "_QC.csv"
        rain_file = "2_Partial_Fill_Daily_RF_mm_1990_2020.csv"
    else:
        pred_file = inputFileList[0]
        temperature_file = inputFileList[1]
        rain_file = inputFileList[2]

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

    if codeIsList:
        ISLAND_code = iCODE
    else:
        ISLAND_code = [iCODE]
        if iCODE == 'MN':
            ISLAND_code = ['MA', 'KO', 'MO', 'LA']

    # importing daily temperature
    df_station = get_Temperature(
        temperature_file,
        meta_columns,
        ISLAND_code,
        mixHighAlt=mixHighAlt)

    # importing average rainfall over the past 180 days with minimum required
    # data for 120 days
    rf_station = get_Rain(
        rain_file,
        meta_columns,
        ISLAND_code,
        window=180,
        min_periods=120, mixHighAlt=mixHighAlt)

    # importing predictior parameters
    pr_station = get_Predictors(pred_file, ISLAND_code)

    return df_station, rf_station, pr_station, ISLAND_code


# In[ ]:


def Gnerate_Metrics(iCODE, params, mode, inversion=2150):

    outFile = 'LIN/LIN_T' + mode + '_' + iCODE + '_'
    for p in params:
        outFile += p + '.linear_'
    outFile += 'params.csv'

    param_List = ["Island", "LON", "LAT"] + params

    df_station, rf_station, pr_station, ISLAND_code = Data_prep(iCODE, mode)

    ###########################################
    Hyper = {}
    Hyper['Date'] = []
    Hyper['MAE'] = []
    Hyper['RMSE'] = []
    Hyper['R2'] = []
    Hyper['AIC'] = []
    Hyper['AICc'] = []
    Hyper['BIC'] = []

    Hyper['MAE_loo'] = []
    Hyper['RMSE_loo'] = []
    Hyper['R2_loo'] = []
    Hyper['AIC_loo'] = []
    Hyper['AICc_loo'] = []
    Hyper['BIC_loo'] = []

    # for t in range(len(param_List)):
    #     Hyper['t' + str(t)] = []
    ###########################################

    dt_cols = df_station.columns[3:][::-1]

    for dt_col in dt_cols:

        try:
            date_str = str(dt_col.date())

            df_date = prep_Data(date_str, df_station, rf_station, pr_station)

            MODEL = myModel(inversion)
            theta, cov, X, y = makeModel(df_date, param_List, MODEL, inversion=inversion)

            if not theta is None:

                y_model = MODEL(X[:, 3:], *theta)
                n_data, n_params = X.shape


                
                u, v = sigma_Clip(y, y_model)
                MAE, RMSE, R2, AIC, AICc, BIC = metrics(
                    u, v, verbose=False, n_param=len(theta), n_data=n_data)

                # leave one out metrics
                MAE_, RMSE_, R2_, AIC_, AICc_, BIC_ = loo_Metrics(X, y, MODEL)

                Hyper['Date'].append(date_str)

                Hyper['MAE'].append(MAE)
                Hyper['RMSE'].append(RMSE)
                Hyper['R2'].append(R2)
                Hyper['AIC'].append(AIC)
                Hyper['AICc'].append(AICc)
                Hyper['BIC'].append(BIC)

                Hyper['MAE_loo'].append(MAE_)
                Hyper['RMSE_loo'].append(RMSE_)
                Hyper['R2_loo'].append(R2_)
                Hyper['AIC_loo'].append(AIC_)
                Hyper['AICc_loo'].append(AICc_)
                Hyper['BIC_loo'].append(BIC_)

                # for t in range(n_params):
                #     Hyper['t' + str(t)].append(theta[t])

                if True:  # len(Hyper['MAE'])%200==0:
                    pd.DataFrame.from_dict(Hyper).to_csv(
                        outFile, sep=',', index=False)
        except BaseException:
            pass
            # print(date_str, len(df_date))

    # finalizing data storage
    pd.DataFrame.from_dict(Hyper).to_csv(outFile, sep=',', index=False)


# In[ ]:

# This code has been automatically covnerted to comply with the pep8 convention
# This the Linux command:
# $ autopep8 --in-place --aggressive  <filename>.py
if __name__ == '__main__':

    iCODE = 'BI'  # str(sys.argv[1])
    mode  = 'max' # str(sys.argv[2])
    #params = ["ELEV.m.", "coastDist", "lai", "tpi", "RF"]

    params = ["ELEV.m."]
    Gnerate_Metrics(iCODE, params, mode)
