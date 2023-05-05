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

from mpl_toolkits.mplot3d import Axes3D
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

data = np.array([[del18],[dust]])[:,0,:].T
matr = data.T @ data
u, s, vh = np.linalg.svd(matr, full_matrices=True)
data_pca = data @ vh.T
x_1 = data_pca[:,0]; x_2 = data_pca[:,1];

kmcs_xy, edges_xy = kmc.km(data_pca, bw=0.31, bins=bins, powers=powers)
X_edge, Y_edge = np.meshgrid(*edges_xy)


# %% ############################## Fig 04 #####################################

fig, ax = plt.subplots(2, 2, figsize=(16,10))

loc = kmcs_xy[0,:,:][:].T

# create the normalisations and levels
norm = cm.colors.Normalize(vmax=loc.max(), vmin=loc.min())
levels = np.linspace(loc.min(),loc.max(), 10)

# create a mask for where the PDF is too small
masker = kmcs_xy[0,...][:].T > .015

# create a copy of loc for the masking.
loc1 = np.copy(loc*masker[:,:])
loc1[loc1==0.] = np.nan

# masker2 is used later for calculating the integrals over D10 and D01.
masker2 = loc1>0.005
ax[0,0].contour(edges_xy[0][:],edges_xy[1][:], loc1, levels=levels, norm=norm,
    origin='lower')
ax[0,0].contourf(edges_xy[0][:],edges_xy[1][:], loc1, levels=levels, norm=norm,
    alpha=0.2, origin='lower')
ax[0,0].contour(edges_xy[0][:],edges_xy[1][:], masker[:,:],
           levels=[1],colors='k', origin='lower', linestyles='dotted');


hist, edge = np.histogram(x_2, bins=30, density=True)
x=(edge[1:] + edge[:-1])/2
ax_hist_v = ax[0,0].inset_axes([1.03,0,.07,1])
ax_hist_v.barh(x, height=np.mean(np.gradient(x))*.8, width=hist,
    label=labels[0], color=colours[0], lw=2, edgecolor='k')
ax_hist_v.set_xlim([0,.95])

hist, edge = np.histogram(x_1, bins=60, density=True)
x=(edge[1:] + edge[:-1])/2
ax_hist_h = ax[0,0].inset_axes([0,.93,1,.1])
ax_hist_h.bar(x, hist, label=labels[1], color=colours[1], lw=2, edgecolor='k',
    width=np.mean(np.gradient(x))*.8)
ax_hist_h.set_xlim([-3,3])

####################################
ax[0,0].text(.29, .51,r'GI', fontsize=28, color='k',
    transform=ax[0,0].transAxes);
ax[0,0].text(.64, .41,r'GS', fontsize=28, color='k',
    transform=ax[0,0].transAxes);
####################################
ax[0,0].plot(x_1[~stadial_mask], x_2[~stadial_mask], 'o', color='#fc8d59',
    alpha=.12)
ax[0,0].plot(x_1[stadial_mask], x_2[stadial_mask], 'o', color='#91bfdb',
    alpha=.12)

loc = -np.cumsum(kmcs_xy[1,:,:].T, axis=1)
loc_temp = np.copy(loc)

# subtract minimum along the vertical axis to all minima are zero
loc = (loc - np.vstack(np.min(loc, axis=1)))

loc1 = np.copy(masker[::15,::15]*kmcs_xy[1,::15,::15].T)
loc1[loc1==0] = np.nan
colouring = np.abs(loc1)

ax[1,1].quiver(edges_xy[0][::15],edges_xy[1][::15],
    loc1, 0, masker[::15,::15] * colouring, scale=7, width=.007, pivot='mid')

ax[1,1].contour(edges_xy[0][:],edges_xy[1][:], masker[:,:],
    levels=[1], colors='k', origin='lower', linestyles='dotted');
minimum = edges_xy[0][:][np.argmin(loc_temp[160:-100,:] \
    * masker[160:-100,:], axis=1)]
ax[1,1].plot(np.where(~(minimum < -2), minimum, np.nan)[:],
             edges_xy[1][160:-100],
             '-', lw=3, color = '#feb24c', label='stable nullcline')
ax[1,0].plot(np.where(~(minimum < -2), minimum, np.nan)[:],
             edges_xy[1][160:-100],
             '-', lw=3, color = '#feb24c')
# ###########################################################################

# use the second masker (masker2) to perform the indefinite integral
loc = -np.cumsum(masker2*kmcs_xy[2,:,:].T, axis=0)

loc_temp = np.copy(loc)

# subtract minimum along the vertical axis to all minima are zero
loc = (loc - np.min(loc, axis=0))

# levels in log scale so they are easier to distinguish
norm = cm.colors.Normalize(vmax=loc.max(), vmin=loc.min())
levels = np.logspace(np.log10(loc.min()+1), np.log10(loc.max()), 20)-1

# copy and mask
loc1 = np.copy(loc*masker[:,:])
loc1[loc1==0] = np.nan

loc1 = np.copy(masker[::15,::15]*kmcs_xy[2,::15,::15].T)
loc1[loc1==0] = np.nan
colouring = np.abs(loc1)

ax[0,1].quiver(edges_xy[0][::15],edges_xy[1][::15], loc1*0, loc1*1,
    masker[::15,::15]*colouring, scale=7, width = .007, pivot='mid')

ax[0,1].contour(edges_xy[0][:],edges_xy[1][:], masker[:,:],
    levels=[1], colors='k', origin='lower', linestyles='dotted');

# Find the minima and maxima
minimum = edges_xy[1][:][np.argmin(loc_temp[:,:]*masker[:,:], axis=0)]

ax[0,1].plot(edges_xy[0][35:-30],
             np.where(~(minimum < -1.25), minimum, np.nan)[35:-30],
             '-', lw=4, color = '#b24cfe', label = 'stable nullcline')
ax[1,0].plot(edges_xy[0][35:-30],
             np.where(~(minimum < -1.25), minimum, np.nan)[35:-30],
             '-', lw=4, color = '#b24cfe', label = 'stable nullcline')

# ###########################################################################

loc1 = np.copy(masker[::15,::15]*kmcs_xy[1,::15,::15].T)
loc2 = np.copy(masker[::15,::15]*kmcs_xy[2,::15,::15].T)

loc1[loc1==0] = np.nan; loc2[loc2==0] = np.nan
colouring = np.sqrt(((loc1)**2 + ((loc2)**2)))

ax[1,0].quiver(edges_xy[0][::15],edges_xy[1][::15],
    loc1, loc2, masker[::15,::15]*colouring, scale=10, width=.007, pivot='mid')

ax[1,0].contour(edges_xy[0][:],edges_xy[1][:], masker[:,:],
    levels=[1], colors='k', origin='lower', linestyles='dotted');

axi_c = fig.add_axes([.95, .08, .005, .4])
cb = matplotlib.colorbar.ColorbarBase(ax=axi_c, cmap=cm.viridis,
                                 orientation="vertical", alpha=.3)
cb.set_ticks(np.array([.0,.2,.4,.6,.8,1.])/1.03)
cb.set_ticklabels(np.array([.0,.2,.4,.6,.8,1.]))

[[ax[i,j].spines['right'].set_visible(False) for i in [0,1]] for j in [0,1]]
[[ax[i,j].spines['top'].set_visible(False) for i in [0,1]] for j in [0,1]]
[ax[i,1].spines['left'].set_visible(False) for i in [0,1]]
[ax[0,j].spines['bottom'].set_visible(False) for j in [0,1]]
[ax[i,0].spines['left'].set_position(('outward', 20)) for i in [0,1]]
[ax[1,j].spines['bottom'].set_position(('outward', 20)) for j in [0,1]]

ax[0,0].text(.85,.02,r'PDF',fontsize=24, ha='right',
    transform=ax[0,0].transAxes);
ax[0,1].text(.95,.02,r'$p_1$ drift $D_{1,0}(p_1,p_2)$',fontsize=24, ha='right',
    transform=ax[0,1].transAxes);
ax[1,1].text(.95,.02,r'$p_2$  drift $D_{0,1}(p_1,p_2)$',fontsize=24, ha='right',
    transform=ax[1,1].transAxes);
ax[1,0].text(.85,.02,r'$\boldsymbol{F}(p_1,p_2)$',fontsize=24, ha='right',
    transform=ax[1,0].transAxes);

ax[0,0].text(.01,.99,r'(a)',fontsize=24, transform=ax[0,0].transAxes);
ax[1,0].text(.01,.99,r'(b)',fontsize=24, transform=ax[1,0].transAxes);
ax[0,1].text(.01,.99,r'(c)',fontsize=24, transform=ax[0,1].transAxes);
ax[1,1].text(.01,.99,r'(d)',fontsize=24, transform=ax[1,1].transAxes);

ax[0,1].legend(loc=1, ncol=1, columnspacing=.6, handlelength=1,
    handletextpad=.5)
ax[1,1].legend(loc=1, ncol=1, columnspacing=.6, handlelength=1,
    handletextpad=.5)

ax_hist_v.set_axis_off()
ax_hist_h.set_axis_off()

ax[0,0].set_ylabel(r'$p_2$')
ax[1,0].set_ylabel(r'$p_2$')
ax[1,1].set_xlabel(r'$p_1$')
ax[1,0].set_xlabel(r'$p_1$')
[ax[i,1].set_yticks([]) for i in [0,1]]
[ax[0,j].set_xticks([]) for j in [0,1]]
[[ax[i,j].set_xlim([-3.3, 3.3]) for i in [0,1]] for j in [0,1]]
[[ax[i,j].set_ylim([-2, 2]) for i in [0,1]] for j in [0,1]]
[ax[1,j].set_xticks([-3.3, -2.2, -1.1, 0, 1.1, 2.2, 3.3]) for j in [0,1]]

fig.subplots_adjust(left=.08, bottom=.10, right=.95, top=.95, wspace=.07,
    hspace=.07)
fig.savefig('../figures/fig04.pdf', dpi=600, transparent = True)
