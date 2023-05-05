# created by Leonardo Rydin Gorjão and Keno Riechers. Most python libraries
# are standard (e.g. via Anaconda). The 'kramersmoyal' library can be installed
# with 'pip install kramersmoyal'. If TeX is not present in the system comment
# out lines 11 to 16.

import numpy as np
import kramersmoyal as kmc

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 18,
    'axes.labelsize': 20,'axes.titlesize': 24, 'figure.titlesize' : 28})
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
matplotlib.rcParams['text.usetex'] = True

from matplotlib import cm
from matplotlib.lines import Line2D

colours = [r'#4daf4a', r'#984ea3']
labels = [r'dust [n.u.]', r'$\delta^{18}$O [n.u.]']

# %% ########################### Load the data #################################
# To generate the data, please see fig01.py. Herein we load the
# pre-detrended data directly.

# Note that the data is reversed in time, thus [::-1]
del18 = np.load('../detrended_data/d18o_temperature_detrended.npy')[::-1]
dust = np.load('../detrended_data/dust_temperature_detrended.npy')[::-1]
stadial_mask = np.load('../detrended_data/stadial_mask.npy')[::-1]

# %% ############################## Kramers--Moyal #############################

timeseries = np.zeros((del18.size, 2))
timeseries[:,0] = del18
timeseries[:,1] = dust

powers = np.array([[0,0], [1,0], [0,1], [1,1], [2,0], [0,2]])
bins = np.array([400, 400])

kmcs_xy, edges_xy = kmc.km(timeseries, bw=0.42, bins=bins, powers=powers)
X_edge, Y_edge = np.meshgrid(*edges_xy)

# %% ############################## Fig 04 #####################################

fig, ax = plt.subplots(2, 2, figsize=(16,8))

loc = kmcs_xy[0,30:-30,30:-30][:].T

# create the normalisations and levels
norm = cm.colors.Normalize(vmax=loc.max(), vmin=loc.min())
levels = np.linspace(loc.min(),loc.max(),10)

# create a mask for where the PDF is too small
masker = kmcs_xy[0,...][:].T>0.015

# create a copy of loc for the masking.
loc1 = np.copy(loc*masker[30:-30,30:-30])
loc1[loc1==0.] = np.nan

masker2 = loc1>0.005

ax[0,0].contour(edges_xy[0][30:-30],edges_xy[1][30:-30], loc1, levels=levels,
    norm=norm, origin='lower')
ax[0,0].contourf(edges_xy[0][30:-30],edges_xy[1][30:-30], loc1, levels=levels,
    norm=norm, alpha=0.2, origin='lower')
ax[0,0].contour(edges_xy[0][30:-30],edges_xy[1][30:-30], masker[30:-30,30:-30],
           levels=[1], colors='k', origin='lower', linestyles='dotted');

hist, edge = np.histogram(dust, bins=30, density=True)
x = ( edge[1:] + edge[:-1] ) / 2
ax_hist_v = ax[0,0].inset_axes([.95,0,0.07,1])
ax_hist_v.barh(x, height=np.mean(np.gradient(x))*0.8, width=hist,
    label=labels[0], color=colours[0], lw=2, edgecolor='k')
ax_hist_v.set_xlim([0,0.55])

hist, edge = np.histogram(del18, bins=60, density=True)
x = ( edge[1:] + edge[:-1] ) / 2
ax_hist_h = ax[0,0].inset_axes([0,.93,1,0.1])
ax_hist_h.bar(x, hist, label=labels[1], color=colours[1], lw=2, edgecolor='k',
    width=np.mean(np.gradient(x))*0.8)
ax_hist_h.set_xlim([-3,3])


ax[0,0].text(.34, .30,r'GS',fontsize=28, color='k', transform=ax[0,0].transAxes)
ax[0,0].text(.56, .62,r'GI',fontsize=28, color='k', transform=ax[0,0].transAxes)

ax[0,0].plot(del18[~stadial_mask],dust[~stadial_mask], 'o', color='#fc8d59',
    alpha=.12)
ax[0,0].plot(del18[stadial_mask],dust[stadial_mask], 'o', color='#91bfdb',
    alpha=.12)

loc = -np.cumsum(kmcs_xy[1,30:-30,30:-30].T, axis=1)
loc_temp = np.copy(loc)

loc = (loc - np.vstack(np.min(loc, axis=1)))

loc1 = np.copy(masker[30:-30:15,30:-30:15] * kmcs_xy[1,30:-30:15,30:-30:15].T)
loc1[loc1==0] = np.nan
colouring = (np.sqrt((loc1)**2))

ax[1,1].quiver(edges_xy[0][30:-30:15],edges_xy[1][30:-30:15],
    loc1, 0, masker[30:-30:15,30:-30:15] * colouring, scale=10,
    width=.007, pivot='mid')

ax[1,1].contour(edges_xy[0][30:-30],edges_xy[1][30:-30], masker[30:-30,30:-30],
    levels=[1], colors='k', origin='lower', linestyles='dotted');

minimum = edges_xy[0][30:-30][np.argmin(loc_temp*masker[30:-30,30:-30], axis=1)]
ax[1,1].plot(np.where(~(minimum < -2), minimum, np.nan)[33:-33],
    edges_xy[1][63:-63], '-', lw=4, color = '#feb24c')
# ax[1,0].plot(np.where(~(minimum < -2), minimum, np.nan)[33:-33],
#     edges_xy[1][63:-63], '-', lw=4, color = '#feb24c')
# ###########################################################################

# use the second masker (masker2) to perform the indefinite integral
loc = -np.cumsum(masker2*kmcs_xy[2,30:-30,30:-30].T, axis=0)
loc_temp = np.copy(loc)

loc = (loc - np.min(loc, axis=0))

norm = cm.colors.Normalize(vmax=loc.max(), vmin=loc.min())

loc1 = np.copy(loc*masker[30:-30,30:-30])
loc1[loc1==0] = np.nan

loc1 = np.copy(masker[30:-30:15,30:-30:15] * kmcs_xy[2,30:-30:15,30:-30:15].T)
loc1[loc1==0] = np.nan
colouring = np.abs(loc1)

ax[0,1].quiver(edges_xy[0][30:-30:15],edges_xy[1][30:-30:15],
    loc1*0., loc1*1., masker[30:-30:15,30:-30:15] * colouring, scale=2.,
    width=.007, pivot='mid')

ax[0,1].contour(edges_xy[0][30:-30],edges_xy[1][30:-30], masker[30:-30,30:-30],
    levels=[1], colors='k', origin='lower', linestyles='dotted');

# Find the minima and maxima
minimum = edges_xy[1][30:-210][np.argmin(loc_temp[:-180,85:] *
    masker[30:-210,115:-30], axis=0)]
ax[0,1].plot(edges_xy[0][115:-40],
    np.where(~(minimum < -1.45), minimum, np.nan)[:-10],
    '-', lw=4, color='#feb24c')
# ax[1,0].plot(edges_xy[0][115:-40],
#     np.where(~(minimum < -1.45), minimum, np.nan)[:-10],
#     '-', lw=4, color='#feb24c')

maximum = edges_xy[1][175:-160][np.argmax(loc_temp[145:-130,:-68] *
    masker[175:-160,30:-98], axis=0)]
ax[0,1].plot(edges_xy[0][188:-98], maximum[158:], '-', lw=4, color='red')
# ax[1,0].plot(edges_xy[0][188:-98], maximum[158:], '-', lw=4, color='red')

minimum2 = edges_xy[1][240:330][np.argmin(loc_temp[210:300,...]
    * masker[240:330,30:-30], axis=0)]
ax[0,1].plot(edges_xy[0][188:-33], np.where(minimum2, minimum2, np.nan)[158:-3],
    '-', lw=4, color = '#feb24c')
# ax[1,0].plot(edges_xy[0][188:-33], np.where(minimum2, minimum2, np.nan)[158:-3],
#     '-', lw=4, color = '#feb24c')

# ###########################################################################

loc1 = np.copy(masker[30:-30:15,30:-30:15]*kmcs_xy[1,30:-30:15,30:-30:15].T)
loc2 = np.copy(masker[30:-30:15,30:-30:15]*kmcs_xy[2,30:-30:15,30:-30:15].T)

loc1[loc1==0] = np.nan; loc2[loc2==0] = np.nan

colouring = np.sqrt(((loc1)**2 + ((loc2)**2)))

ax[1,0].quiver(edges_xy[0][30:-30:15],edges_xy[1][30:-30:15], loc1, loc2,
    masker[30:-30:15,30:-30:15]*colouring, scale=7, width = 0.007, pivot='mid')

ax[1,0].contour(edges_xy[0][30:-30],edges_xy[1][30:-30], masker[30:-30,30:-30],
    levels=[1], colors='k', origin='lower', linestyles='dotted');

axi_c = fig.add_axes([.95, .1, .005, .4])
cb = matplotlib.colorbar.ColorbarBase(ax=axi_c, cmap=cm.viridis,
                                 orientation="vertical", alpha=.3)
cb.set_ticks(np.array([.0, .2, .4, .6, .8, 1.]) / 1.03)
cb.set_ticklabels(np.array([.0, .2, .4, .6, .8, 1.]))

[[ax[i,j].spines['right'].set_visible(False) for i in [0,1]] for j in [0,1]]
[[ax[i,j].spines['top'].set_visible(False) for i in [0,1]] for j in [0,1]]
[ax[i,1].spines['left'].set_visible(False) for i in [0,1]]
[ax[0,j].spines['bottom'].set_visible(False) for j in [0,1]]
[ax[i,0].spines['left'].set_position(('outward', 20)) for i in [0,1]]
[ax[1,j].spines['bottom'].set_position(('outward', 20)) for j in [0,1]]

ax[0,0].text(0.85, .02,r'PDF',fontsize=24, ha='right',
    transform=ax[0,0].transAxes);
ax[0,1].text(0.95, .02,
    r'dust-drift $D_{1,0}(\delta^{18}\mathrm{O},\mathrm{dust})$',
    fontsize=24, ha='right', transform=ax[0,1].transAxes);
ax[1,1].text(0.95, .02,
    r'$\delta^{18}$O-drift $D_{0,1}(\delta^{18}\mathrm{O},\mathrm{dust})$',
    fontsize=24, ha='right', transform=ax[1,1].transAxes);
ax[1,0].text(0.85,.02,r'$\boldsymbol{F}(\delta^{18}\mathrm{O},\mathrm{dust})$', fontsize=24, ha='right',
    transform=ax[1,0].transAxes);

ax[0,0].text(0.01,0.99, r'(a)', fontsize=24, transform=ax[0,0].transAxes);
ax[1,0].text(0.01,0.99, r'(b)', fontsize=24, transform=ax[1,0].transAxes);
ax[0,1].text(0.01,0.99, r'(c)', fontsize=24, transform=ax[0,1].transAxes);
ax[1,1].text(0.01,0.99, r'(d)', fontsize=24, transform=ax[1,1].transAxes);

L = [Line2D([0],[0], lw=4, color='#feb24c', label='stable nullcline'),
     Line2D([0],[0], lw=4, color='red', label='unstable nullcline'),
     Line2D([0],[0], lw=0, ms=15, marker='X', color='#feb24c',
        label='stable fixed point'),
     Line2D([0],[0], lw=0, ms=15, marker='X', color='red',
        label='unstable fixed point')]
leg0 = ax[0,1].legend(handles=L,loc=2, ncol=1, columnspacing=0.6,
    handlelength=0.7, handletextpad=0.3, bbox_to_anchor=(-.17,-.1))

ax[0,1].add_artist(leg0)

ax_hist_v.set_axis_off()
ax_hist_h.set_axis_off()

ax[0,0].set_ylabel(labels[0])
ax[1,0].set_ylabel(labels[0])
ax[1,1].set_xlabel(labels[1])
ax[1,0].set_xlabel(labels[1])
[ax[i,1].set_yticks([]) for i in [0,1]]
[ax[0,j].set_xticks([]) for j in [0,1]]
[[ax[i,j].set_xlim([-3, 3]) for i in [0,1]] for j in [0,1]]
[[ax[i,j].set_ylim([-3, 3]) for i in [0,1]] for j in [0,1]]
[ax[1,j].set_xticks([-3, -2, -1, 0, 1, 2, 3]) for j in [0,1]]

ax[1,0].plot(-.75,-1.05, marker='X', ms=15, color='#feb24c');
ax[1,0].plot(-.25,-.1, marker='X', ms=15, color='red');
ax[1,0].plot(.7,.95, marker='X', ms=15, color='#feb24c');

fig.subplots_adjust(left=.08, bottom=.12, right=.95, top=.96, wspace=.07,
    hspace=.07)
fig.savefig('../figures/fig03.pdf', dpi=600, transparent=True)
