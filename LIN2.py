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

from Temp_linear import *


if __name__ == '__main__':

    iCODE = str(sys.argv[1])
    mode = str(sys.argv[2])

    params = ["ELEV.m.", "coastDist", "lai", "tpi"]
    Gnerate_Metrics(iCODE, params, mode)

    params = ["ELEV.m.", "coastDist", "lai"]
    Gnerate_Metrics(iCODE, params, mode)

    params = ["ELEV.m.", "coastDist", "lai", "windSpeed"]
    Gnerate_Metrics(iCODE, params, mode)

    params = ["ELEV.m.", "coastDist", "lai", "RF"]
    Gnerate_Metrics(iCODE, params, mode)

    params = ["ELEV.m.", "coastDist", "lai", "albedo"]
    Gnerate_Metrics(iCODE, params, mode)

    params = ["ELEV.m.", "coastDist", "lai", "albedo", "windSpeed"]
    Gnerate_Metrics(iCODE, params, mode)
