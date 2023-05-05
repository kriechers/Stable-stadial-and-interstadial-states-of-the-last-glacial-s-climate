import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %matplotlib inline
import matplotlib
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
# plt.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 18,
    'axes.labelsize': 20,'axes.titlesize': 24, 'figure.titlesize' : 28})
matplotlib.rcParams['text.usetex'] = True
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

colours = ['#4daf4a','#984ea3','#e78ac3','#377eb8','#ff7f00','#e41a1c']
labels  = [r'dust [n.u.]', r'$\delta^{18}$O [n.u.]']

################################ Load the data #################################
# %% Read the data

del18 = np.load('detrended_data/d18o_temperature_detrended.npy')[::-1]
dust = np.load('detrended_data/dust_temperature_detrended.npy')[::-1]
ts = [del18, dust]

# %% package with the tests
from arch.unitroot import ADF, DFGLS
trends = (['n', 'c', 'ct', 'ctt'], ['c','ct'])

# find critical values. They are all the same since the time series have (almost) the same length
cv_ADF = [[np.round(ADF(ele, trend = r).critical_values['5%'], 6) for r in trends[0]] for ele in ts]
cv_DFGLS = [[np.round(DFGLS(ele, trend = r).critical_values['5%'], 6) for r in trends[1]] for ele in ts]

# %%
ADF_opt = np.vstack(np.array([[str(np.round(ADF(ele, trend=r, lags=10).stat, 4))
    + ' (' + str(np.format_float_scientific(ADF(ele, trend=r, lags=10).pvalue, precision = 3)) + ')'
    + ' [' + str(ADF(ele, trend=r, lags=10).lags) + ']' for r in trends[0]] for ele in ts]))
DFGLS_opt = np.vstack(np.array([[str(np.round(DFGLS(ele, trend=r, lags=10).stat, 4))
    + ' (' + str(np.format_float_scientific(DFGLS(ele, trend=r, lags=10).pvalue, precision = 3)) + ')'
    + ' [' + str(DFGLS(ele, trend=r, lags=10).lags) + ']' for r in trends[1]] for ele in ts]))

# %%
# to see each table, just use pandas DataFrame around it, it just looks nicer
pd.DataFrame(ADF_opt, columns=trends[0], index=['del18', 'dust'])
pd.DataFrame(DFGLS_opt, columns=trends[1], index=['del18', 'dust'])


# %%
lags = np.linspace(1,100,100, dtype=int)

ADF_opt_lags_del18 = np.vstack(np.array([[str(np.round(ADF(del18, trend=r, lags=l).stat, 4))
    + ' (' + str(np.format_float_scientific(ADF(del18, trend=r, lags=l).pvalue, precision = 3)) + ')'
    + ' [' + str(ADF(del18, trend=r, lags=l).lags) + ']' for r in trends[0]] for l in lags]))
ADF_opt_lags_dust = np.vstack(np.array([[str(np.round(ADF(dust, trend=r, lags=l).stat, 4))
    + ' (' + str(np.format_float_scientific(ADF(dust, trend=r, lags=l).pvalue, precision = 3)) + ')'
    + ' [' + str(ADF(dust, trend=r, lags=l).lags) + ']' for r in trends[0]] for l in lags]))

DFGLS_opt_lags_del18 = np.vstack(np.array([[str(np.round(DFGLS(del18, trend=r, lags=l).stat, 4))
    + ' (' + str(np.format_float_scientific(DFGLS(del18, trend=r, lags=l).pvalue, precision = 3)) + ')'
    + ' [' + str(DFGLS(del18, trend=r, lags=l).lags) + ']' for r in trends[1]] for l in lags]))
DFGLS_opt_lags_dust = np.vstack(np.array([[str(np.round(DFGLS(dust, trend=r, lags=l).stat, 4))
    + ' (' + str(np.format_float_scientific(DFGLS(dust, trend=r, lags=l).pvalue, precision = 3)) + ')'
    + ' [' + str(DFGLS(dust, trend=r, lags=l).lags) + ']' for r in trends[1]] for l in lags]))

pd.DataFrame(ADF_opt_lags_del18, columns=trends[0])
pd.DataFrame(ADF_opt_lags_dust, columns=trends[0])
pd.DataFrame(DFGLS_opt_lags_del18, columns=trends[1])
pd.DataFrame(DFGLS_opt_lags_dust, columns=trends[1])

# %%
lags = np.linspace(1,100,100, dtype=int)

ADF_opt_lags_del18_boolean = np.vstack(np.array([[ADF(del18, trend=r, lags=l).stat < cv_ADF[0][i] for i, r in enumerate(trends[0])] for l in lags]))
ADF_opt_lags_dust_boolean = np.vstack(np.array([[ADF(dust, trend=r, lags=l).stat < cv_ADF[1][i] for i, r in enumerate(trends[0])] for l in lags]))

DFGLS_opt_lags_del18_boolean = np.vstack(np.array([[DFGLS(del18, trend=r, lags=l).stat < cv_DFGLS[0][i] for i, r in enumerate(trends[1])] for l in lags]))
DFGLS_opt_lags_dust_boolean = np.vstack(np.array([[DFGLS(dust, trend=r, lags=l).stat < cv_DFGLS[1][i] for i, r in enumerate(trends[1])] for l in lags]))
