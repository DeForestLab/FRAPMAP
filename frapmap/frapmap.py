import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import csv
from scipy.special import i0, ive
from scipy.optimize import curve_fit

import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette('viridis')

def fit_recovery(time, fluor, w, mode='bessel'):
    """
    Fits FRAP recovery curves to solve for the diffusion coefficient and max recovery value.
    Equations derived from DeForest and Tirrell, 2015.
    Allowing "a" to vary helps with fitting partial recovery curves or for systems where full recovery is not expected.

    Parameters
    ----------
    time : array-like
        Time points of the FRAP experiment.
    fluor : array-like
        Fluorescence recovery values corresponding to the time points.
    w : float
        Radius of the photobleached spot. Note that the final diff value will have the same units as w.
    mode : str, optional
        Fitting mode to use. Currently only 'bessel' is implemented.

    Returns
    -------
    diff : float
        Diffusion coefficient in (units of w)^2/s.
    a : float
        Correction for max fluorescence at complete recovery.

    """


    if time[0] != 0:
       # Normalize time
       time = np.array([i-np.min(time) for i in time][1:])

    if np.max(fluor) != 1:
       # Normalize fluorescence
       fluor = np.array([(i-np.min(fluor))/(np.max(fluor)-np.min(fluor)) for i in fluor][1:])

    if mode == 'bessel':
        # Define function to fit
        func = lambda var,td,a : a*np.exp(-td/(2*var)) * (i0(td/(2*var)) + ive(1, td/(2*var)))
        [td,a], pcov = curve_fit(func, time, fluor, diag=(1./time.mean(),1./fluor.mean()))
        diff = w**2/td
    
    else:
       print('Other solvers not supported yet.')
       diff = 0    

    return diff, a

def calcG(Df, k, a, b):
  '''
    Simple exponential for correlating diffusivity and storage modulus. Requires a previously fit standard curve to obtain k, a, and b.
    Df must be in m^2/s in order to return g in Pascals.

    Parameters
    ----------
    Df: float
      Diffusivity in m^2/s
    k: float
      Fitting parameter
    a: float
      Fitting parameter
    b: float
      Fitting parameter

    Returns
    -------
    g: float
      Storage modulus in Pa
  '''
  g = np.exp(-k*Df + b) + a
  return g

def batch_csv(path, w, order_on='[0-9]+', time_col=0, fluor_col=1, encode='utf-16'):
    '''
    Batch process multiple csv files to obtain diffusion coefficients in a dataframe.
    '''
    diffs = []
    a_s = []
    orders = []

    for file in os.listdir(path):
        if file.endswith('.csv'):
            
            order = int(re.search(order_on, file).group())
            
            time = []
            fluor = []

            with open(path+file, encoding=encode, errors='ignore') as f:
                r = csv.reader(f, delimiter=',')
                for i, row in enumerate(r):
                    time.append(row[time_col])
                    fluor.append(row[fluor_col])
            
            time = [np.float64(i) for i in time]
            fluor = [np.float64(i) for i in fluor]
            diff, a = fit_recovery(time, fluor, w)

            diff, a = fit_recovery(time, fluor, w)

            diffs.append(diff)
            a_s.append(a)
            orders.append(order)

    df = pd.DataFrame({'diffusion_coefficient': diffs, 'a': a_s, 'order': orders})

    df.sort_values(by='order', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

# def batch_csv_spatial(path, w, order_on='[0-9]+', time_col=0, fluor_col=1):

#    diffs = []
#    a_s = []
#    orders = []

#    for file in os.listdir(path):
#        if file.endswith('.csv'):
           
#             coord = file.split('-')

#             order = [coord[i] for i in range(1, len(coord)-1)]

#             if len(order) == 1:
#                 return batch_csv(path, w, order_on=order_on, time_col=time_col, fluor_col=fluor_col)
            
#             elif len(order) == 2:
#                 pass
#             elif len(order) == 3:
#                 pass
#             else:
#                 raise ValueError('File naming convention not recognized. Ensure files are named as "prefix-X-Y-Z.csv" where X, Y, and Z are spatial coordinates.')

#             data = pd.read_csv(os.path.join(path, file))
#             diff, a = fit_recovery(data.iloc[:, time_col], data.iloc[:, fluor_col], w)

#             diffs.append(diff)
#             a_s.append(a)
#             orders.append(order)

#    df = pd.DataFrame({'diffusion_coefficient': diffs, 'a': a_s, 'order': orders})

#    df.sort_values(by='order', inplace=True)

#    return df
