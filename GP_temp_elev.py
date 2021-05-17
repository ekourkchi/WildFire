import sys, os
import matplotlib.pyplot as plt
import requests
import json
import pandas as pd
from pandas.io import sql
from pandas.io.json import json_normalize
import numpy as np
from datetime import date, timedelta
from datetime import datetime
import time
import re
import pylab as py
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from scipy.optimize import curve_fit
import george
from george import kernels
####################################################################
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

    mse = np.mean((y1-y2)**2)

    RMSE = np.sqrt(mse)
    MAE = np.mean(np.abs(y1-y2))
    R2 = np.max([r2_score(y1, y2),r2_score(y2, y1)])
    
    BIC = calculate_bic(n, mse, n_param)
    AIC = calculate_aic(n, mse, n_param)

    if verbose:
        print('MAE: %.2f'%MAE, ' RMSE: %.2f'%RMSE, ' R^2: %.2f'%R2)
        print('AIC: %.2f'%AIC, ' BIC: %.2f'%BIC)

    return MAE, RMSE, R2, AIC, BIC
####################################################################
def linear(x, a, b):
    return a*x+b

def bilinear(x, a, b, c):
    
    left  = a*x+b
    right = c*(x-2150) + (a*2150+b) 
    
    try:
        y = np.asarray([left[i] if x[i]<=2150 else right[i] for i in range(len(x))])
        return y
    except:
        if x<=2150:
            return left
        else:
            return right

        
def ourliers(X, y):
    
    # plt.plot(X[:,2], y, 'o', mfc='white')
    u = np.arange(np.round(np.max(X[:,2])))

    fit, cov = curve_fit(linear, X[:,2], y, sigma=y*0+1)
    v1 = linear(u, fit[0], fit[1])

    fit, cov = curve_fit(bilinear, X[:,2], y, sigma=y*0+1)



    model = bilinear(X[:,2], fit[0], fit[1], fit[2])

    indx, = np.where(np.abs(model-y)<3)

    fit, cov = curve_fit(bilinear, X[:,2][indx], y[indx], sigma=y[indx]*0+1)


    model = bilinear(X[:,2], fit[0], fit[1], fit[2])

    indx, = np.where(np.abs(model-y)<3)

    fit, cov = curve_fit(bilinear, X[:,2][indx], y[indx], sigma=y[indx]*0+1)

    # plt.plot(u, v1)

    v2 = bilinear(u, fit[0], fit[1], fit[2])
#     plt.plot(u, v2)

    # plt.plot(X[:,2][indx], y[indx], '.')
    # print(fit)
    
    
    lin_model = bilinear(X[:,2][indx], fit[0], fit[1], fit[2])
    data = y[indx]

    MAE, RMSE, R2, AIC, BIC = metrics(lin_model, data, verbose=False, n_param=3)
    return indx, fit[0], fit[1], fit[2], MAE, RMSE, R2, AIC, BIC
####################################################################
def esnModel(theta, X_train, X_test, y_train, y_test, Hinv=2150, biLinear=False):
    
        zp = theta[0]
        slope = theta[1]
        l1 = np.exp(theta[2])
        l2 = np.exp(theta[3])
        sigma = np.exp(theta[4])
        err = np.exp(theta[5])
        
        if biLinear:
            s2 = theta[6]
        

        y_model = slope*X_train[:,2]+zp
        
        if biLinear:
            ind, = np.where(X_train[:,2]>Hinv)
            y_model[ind] += s2*(X_train[:,2][ind]-Hinv)

        kernel = sigma * kernels.ExpSquaredKernel([l1 , l2], ndim=2)
        gp = george.GP(kernel)
        gp.compute(X_train[:,:2], err)

        y_res_test, cov = gp.predict(y_train-y_model, X_test[:,:2], return_var=True)

        y_model_test = slope*X_test[:,2]+zp
        
        if biLinear:
            ind, = np.where(X_test[:,2]>Hinv)
            y_model_test[ind] += s2*(X_test[:,2][ind]-Hinv)    
        
        
        return y_res_test+y_model_test, cov
####################################################################
def nll_fn2(X, y, biLinear=False):
    def step(theta):
        loo = LeaveOneOut()
        XI2 = 0
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
            model_out, cov = esnModel(theta, X_train, X_test, y_train, y_test, biLinear=biLinear)
            XI2 += np.sum((model_out - y_test)**2)
        return XI2
    return step


def solver(X, y, indx, biLinear=False, n_iter=5):
    
    j_iter = 0
    T0 = 0.
    output = []
    while j_iter<n_iter:

        try:
            # Maximum Likelihood
            
            if biLinear:
                res = minimize(nll_fn2(X[indx], y[indx], biLinear=biLinear), [T0, -0.001, 1, 1, 0.1, -1, -0.001], method='SLSQP', 
                          bounds=((None,None), (-0.1, 0), 
                                  (None, 20), (None,20), (None, 20), (-10,1), (-0.1, 0.1)))

            else: 
                res = minimize(nll_fn2(X[indx], y[indx], biLinear=biLinear), [T0, -0.001, 1, 1, 0.1, -1], method='SLSQP', 
                          bounds=((None,None), (-0.1, 0), 
                                  (None, 20), (None,20), (None, 20), (-10,1)))


            theta = res.x
            loo = LeaveOneOut()

            X_ = X[indx]
            y_ = y[indx]

            u = []
            v = []

            for train_index, test_index in loo.split(X_):
                X_train, X_test = X_[train_index], X_[test_index]
                y_train, y_test = y_[train_index], y_[test_index]

                model_out, cov = esnModel(theta, X_train, X_test, y_train, y_test, biLinear=biLinear)  

                u.append(model_out[0])
                v.append(y_test[0])
                
            R2_1 = r2_score(np.asarray(u), np.asarray(v))
            R2_2 = r2_score(np.asarray(v), np.asarray(u))
            
            R2 = np.max([R2_1, R2_2])

            output.append([R2, theta])

            T0 = np.random.uniform(0, 50)
#             print(iter, T0, '%.3f'%R2)
            j_iter+=1
        except:
            T0 = np.random.uniform(0, 50)
#             print(iter, T0)
            continue
    
    return output
####################################################################

if __name__ == '__main__':

	inFile = sys.argv[1]
	island = sys.argv[2]

	df = pd.read_csv(inFile, encoding = "ISO-8859-1", engine='python')

	meta_columns = ['SKN', 'Station.Name', 'Observer', 'Network', 'Island', 'ELEV.m.',
       'LAT', 'LON', 'NCEI.id', 'NWS.id', 'NESDIS.id', 'SCAN.id',
       'SMART_NODE_RF.id']
	Temp_columns = df.columns[13:]


	df2 = df[meta_columns]
	df2 = df2.set_index("SKN")
	df2 = df2[(df2.Island==island)]

	df1 = df[["SKN"]+list(Temp_columns)].T
	new_header = df1.iloc[0]
	df1 = df1[1:] 
	df1.columns = new_header 
	df1.index = pd.to_datetime([x.split('X')[1] for x in df1.index.values])
	df1.index.name = 'Date'
	df1 = df1[list(df2.index.values)]

	df3 = df2[["LON", "LAT", "ELEV.m."]].T
	df3 = df3[list(df2.index.values)]

	df_station =  df3.T
	df_station = df_station.join(df1.T, how='left')

	Hyper = {}
	Hyper['date'] = []
	Hyper['zp'] = []
	Hyper['slope'] = []
	Hyper['l1'] = []
	Hyper['l2'] = []
	Hyper['sigma'] = []
	Hyper['error'] = []
	Hyper['S2'] = []

	Hyper['MAE'] = []
	Hyper['RMSE'] = []
	Hyper['R2'] = []
	Hyper['AIC'] = []
	Hyper['BIC'] = []

	Hyper['a'] = []
	Hyper['b'] = []
	Hyper['c'] = []

	Hyper['MAE_lin'] = []
	Hyper['RMSE_lin'] = []
	Hyper['R2_lin'] = []
	Hyper['AIC_lin'] = []
	Hyper['BIC_lin'] = []

	df_station =  df3.T
	df_station = df_station.join(df1.T, how='left')

	dt_cols = df_station.columns[3:][::-1]

	t1 =  datetime.now()
	###########################################
	for dt_col in dt_cols:
	    
	    try:
	        date_str = str(dt_col.date())

	        Date = pd.to_datetime(date_str)

	        data = df_station[["LON", "LAT", "ELEV.m.", Date]].dropna()
	        X = data[["LON", "LAT", "ELEV.m."]].values
	        SKN = data.index.values
	        y = data[Date].values
	        

	        if len(data) > 1:
	            
	            Hmax = np.max(X[:,2])
	            if Hmax>2150:
	                biLinear=True
	                n_param=7
	                linn_param = 3
	            else:
	                biLinear=False
	                n_param=6
	                linn_param = 2

	            indx, a, b, c, MAE_lin, RMSE_lin, R2_lin, AIC_lin, BIC_lin  = ourliers(X, y)
	            output = solver(X, y, indx, biLinear=biLinear)

	            output.sort(key= lambda x: x[0], reverse=True)

	            theta = output[0][1]
	            R2 = output[0][0]

	            colName = str(Date.date())+'-M'

	            loo = LeaveOneOut()

	            X_ = X[indx]
	            y_ = y[indx]

	            u = y_*0+np.nan
	            u_e = y_*0+np.nan

	            for train_index, test_index in loo.split(X_):
	                X_train, X_test = X_[train_index], X_[test_index]
	                y_train, y_test = y_[train_index], y_[test_index]

	                model_out, cov = esnModel(theta, X_train, X_test, y_train, y_test, biLinear=biLinear)  
	                u[test_index] = model_out
	                u_e[test_index] = np.sqrt(cov)

	            m = y*0+np.nan
	            m_e = y*0+np.nan
	            m[indx] = u
	            m_e[indx] = u_e
	            m[np.abs(m-y)>10]=np.nan

	            data[colName] = m
	            data[colName+'_err'] = m_e

	            df_station = df_station.join(data[[colName, colName+'_err']], how='left')
	            df_station.reset_index().to_csv(inFile.split('.csv')[0]+'_'+island+'.csv', sep=',', index=False)

	            Hyper['date'].append(date_str)
	            Hyper['zp'].append(theta[0])
	            Hyper['slope'].append(theta[1])
	            Hyper['l1'].append(theta[2])
	            Hyper['l2'].append(theta[3])
	            Hyper['sigma'].append(theta[4])
	            Hyper['error'].append(theta[5])
	            
	            if biLinear:
	                Hyper['S2'].append(theta[6])
	            else:
	                Hyper['S2'].append(np.nan)
	            
	            
	            y1 = y[indx]
	            y2 = m[indx]

	            ind, = np.where(np.abs(y1-y2)<10)
	            u = y1[ind]
	            v = y2[ind]

	            MAE, RMSE, R2, AIC, BIC = metrics(u, v, verbose=False, n_param=n_param)
	            
	            
	            # To make sure that the linear model is excuted using the same data
	            lin_model = bilinear(X[:,2][indx][ind], a, b, c)
	            data = y[indx][ind]
	            MAE_lin, RMSE_lin, R2_lin, AIC_lin, BIC_lin = metrics(lin_model, data, 
	                                                                  verbose=False, n_param=linn_param)
	            

	            Hyper['MAE'].append(MAE)
	            Hyper['RMSE'].append(RMSE)
	            Hyper['R2'].append(R2)
	            Hyper['AIC'].append(AIC)
	            Hyper['BIC'].append(BIC)
	            
	            Hyper['a'].append(a)
	            Hyper['b'].append(b)
	            Hyper['c'].append(c)
	            
	            Hyper['MAE_lin'].append(MAE_lin)
	            Hyper['RMSE_lin'].append(RMSE_lin)
	            Hyper['R2_lin'].append(R2_lin)
	            Hyper['AIC_lin'].append(AIC_lin)
	            Hyper['BIC_lin'].append(BIC_lin)
	            
	            pd.DataFrame.from_dict(Hyper).to_csv(inFile.split('.csv')[0]+'_'+island+'_hyper.csv', sep=',', index=False)

	            print(date_str)
	    except:
	        pass
	    

	###########################################
	t2 =  datetime.now()
	print("Execution time:")
	print(t2-t1)
	    







	




