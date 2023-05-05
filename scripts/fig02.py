# created by Leonardo Rydin Gorjão and Keno Riechers. Most python libraries
# are standard (e.g. via Anaconda). The 'kramersmoyal' library can be installed
# with 'pip install kramersmoyal'. If TeX is not present in the system comment
# out lines 8 to 13.

import numpy as np

import matplotlib
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 18,
  'axes.labelsize': 20,'axes.titlesize': 24, 'figure.titlesize' : 28})
matplotlib.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt

colours = [r'#4daf4a', r'#984ea3']
labels = [r'dust [n.u.]', r'$\delta^{18}$O [n.u.]']

# %% ########################### Load the data #################################
# To generate the data, please see fig01.py. Herein we load the
# pre-detrended data directly.

# Note that the data is reversed in time, thus [::-1]
del18 = np.load('../detrended_data/d18o_temperature_detrended.npy')[::-1]
dust = np.load('../detrended_data/dust_temperature_detrended.npy')[::-1]

################################ Start of Plots ################################
# %%
del18_incs = del18[1:] - del18[:-1]
dust_incs = dust[1:] - dust[:-1]

_ = np.correlate(del18_incs - np.mean(del18_incs),
    del18_incs - np.mean(del18_incs), 'full')
del18_incs_corr = _[_.size//2:]/_.max()
_ = np.correlate(dust_incs - np.mean(dust_incs),
    dust_incs - np.mean(dust_incs), 'full')
dust_incs_corr = _[_.size//2:]/_.max()


# %%
fig, ax = plt.subplots(1,1, figsize=(7,3.5));

ax.plot(del18_incs_corr[:12], label=labels[0], color=colours[0], lw=3,
    clip_on=False)
ax.plot(dust_incs_corr[:12], label=labels[1], color=colours[1], lw=3,
    clip_on=False)

ax.set_ylabel(r'$\rho(\tau)$',fontsize=24)
ax.set_xlabel(r'$\tau$')

ax.set_ylim([-.5, 1])
ax.set_yticks([1, 0.5, 0, -.5])
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_xticklabels([0, 2*5, 4*5, 6*5, 8*5, 10*5])
ax.set_xlim([0, 12])

ax.legend(loc=1, ncol=1, columnspacing=.6, handlelength=.5, handletextpad=.2)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_position(('outward', 18))
ax.spines['bottom'].set_position(('outward', 18))

fig.subplots_adjust(left=.18, bottom=.26, right=.98, top=.97, hspace=.28,
    wspace=.105)
fig.savefig('../figures/fig02.pdf', dpi=600, transparent = True)
