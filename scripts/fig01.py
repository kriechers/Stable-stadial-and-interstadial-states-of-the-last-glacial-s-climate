#################################################################
# this script generates fig01 and saves the detrended data      # 
#################################################################

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from scipy.interpolate import interp1d
from functions import binning, NGRIP_stadial_mask, make_patch_spines_invisible, detrend_temperature
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.tsa.stattools import kpss


matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 18,
                            'axes.labelsize': 20, 'axes.titlesize': 24, 'figure.titlesize': 28})
matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}']
matplotlib.rcParams['text.usetex'] = True
d18o_label = r'$\delta^{18}$O [n.u.]'
dust_label = r'dust [n.u.]'
d18o_color = '#984ea3'
dust_color = '#4daf4a'

#################################################################
# import NGRIP d18o and dust 20y mean data (up to 120kyr b2k)   #
#################################################################

d18o_120_data = pd.read_excel('../data/GICC05modelext_GRIP_and_GISP2_and'
                              + '_resampled_data_series_Seierstad_et_al.'
                              + '_2014_version_10Dec2014-2.xlsx',
                              sheet_name='3) d18O and Ca 20 yrs mean',
                              header=None,
                              skiprows=range(52),
                              names=['age', 'd18o'],
                              usecols='A,E')
d18o_120_data.drop_duplicates(subset='age', inplace=True)
d18o_120_data.reset_index(drop=True, inplace=True)
d18o_120_data.dropna(inplace=True)
d18o_120 = d18o_120_data['d18o'].values
d18o_120_time = d18o_120_data['age'].values

dust_120_data = pd.read_table('../data/NGRIP_dust_on_GICC05_20y_december2014.txt',
                              skiprows=56,
                              header=None,
                              names=['depth', 'age', 'MCE', 'dust'])

dust_120_data.drop_duplicates(subset='age', inplace=True)
dust_120_data.reset_index(drop=True, inplace=True)
dust_120_data.dropna(inplace=True, subset=['dust'])
dust_120 = -np.log(dust_120_data['dust'].values)
dust_120_time = dust_120_data['age'].values
#dust_120 = -(dust_120 - np.nanmean(dust_120))/np.nanstd(dust_120)
#d18o_120_data = d18O_120_data.values

#################################################################
# import temperature reconstruction                             #
#################################################################

temp_source = '../data/41586_2016_BFnature19798_MOESM245_ESM.xlsx'

temperature = pd.read_excel(temp_source,
                            usecols=[0, 2],
                            skiprows=1,
                            names=['age', 'T_anom'])
time2temp = interp1d(temperature['age']*1000, temperature['T_anom'])
temp_i = time2temp(d18o_120_time)
#temp_i = (temp_i - np.mean(temp_i))/np.std(temp_i)

#################################################################
# import detrended and normalized data                          # 
#################################################################

# d18o = np.load('../detrended_data/d18o_temperature_detrended.npy')
# dust = np.load('../detrended_data/dust_temperature_detrended.npy')
# time = np.load('../detrended_data/time_temperature_detrended.npy')

#################################################################
# import NGRIP d18o 5cm data (up to 60kyr b2k)                  #
#################################################################

NGRIP_data = pd.read_excel('../data/NGRIP_d18O_and_dust_5cm.xls',
                           sheet_name='NGRIP-2 d18O and Dust',
                           header=0,
                           names=['depth', 'd18o', 'dust', 'age', 'MCE'])

#################################################################
# crop 5cm d18o data                                            #
#################################################################

t_i = 27000
t_f = 59000
NGRIP_data.where(((t_i < NGRIP_data['age']) &
                  (t_f > NGRIP_data['age'])),
                 inplace=True)

d18o = NGRIP_data['d18o'].dropna().values
d18o_time = NGRIP_data.dropna(axis = 0,
                              subset = ['d18o'])['age'].values

#################################################################
# import NGRIP dust 5cm data (up to 60kyr b2k)                  #
#################################################################

dust = -np.log(NGRIP_data['dust'].dropna().values)
dust_time = NGRIP_data.dropna(axis = 0,
                              subset = ['dust'])['age'].values


#################################################################
# detrend the d18o 5cm data                                     #
#################################################################

fig, ax = plt.subplots(2,1)

d18o_detrended, temp_d18o, GI_d18o, GS_d18o, d18o_par = (
    detrend_temperature(d18o, d18o_time, ax[0]))

dust_detrended, temp_dust, GI_dust, GS_dust, dust_par = (
    detrend_temperature(dust, dust_time, ax[1]))

#################################################################
# binning the detrended data                                    #
#################################################################

d18o_binned_time, d18o_binned_detrended = binning(d18o_time,
                                                  d18o_detrended, 
                                                  np.arange(t_i, t_f, 5))

dust_binned_time, dust_binned_detrended = binning(dust_time,
                                                  dust_detrended, 
                                                  np.arange(t_i, t_f, 5))


#################################################################
# linearly interpolate missing data                             #
#################################################################

mask = np.isnan(d18o_binned_detrended)
d18o_fill_nans = interp1d(d18o_binned_time[~mask], d18o_binned_detrended[~mask])
d18o_binned_detrended[mask]= d18o_fill_nans(d18o_binned_time[mask])

mask = np.isnan(dust_binned_detrended)
dust_fill_nans = interp1d(dust_binned_time[~mask], dust_binned_detrended[~mask])
dust_binned_detrended[mask]= dust_fill_nans(dust_binned_time[mask])

#################################################################
# normalizing the data                                          #
#################################################################

d18o_binned_detrended_norm = ((d18o_binned_detrended
                              - np.nanmean(d18o_binned_detrended))
                              /np.nanstd(d18o_binned_detrended))

dust_binned_detrended_norm = ((dust_binned_detrended
                              - np.nanmean(dust_binned_detrended))
                              /np.nanstd(dust_binned_detrended))

#################################################################
# save the data                                                 #
#################################################################

d18o_name = '../detrended_data/d18o_temperature_detrended.npy'
dust_name = '../detrended_data/dust_temperature_detrended.npy'
time_name = '../detrended_data/time_temperature_detrended.npy'

np.save(d18o_name, d18o_binned_detrended_norm)
np.save(dust_name, dust_binned_detrended_norm)
np.save(time_name, dust_binned_time)

#################################################################
# apply ADF test for stationarity                               #
# H0: the data is non-stationry                                 # 
# H1: the data is stationary                                    #
#################################################################

d18o_adf = adf(d18o_binned_detrended_norm)
d18o_kpss = kpss(d18o_binned_detrended_norm, regression = 'ct', nlags = 100)

dust_adf = adf(dust_binned_detrended_norm)
dust_kpss = kpss(dust_binned_detrended_norm,regression = 'ct', nlags = 100)


#################################################################
# visualizations                                                #
#################################################################

#################################################################
# plot the binned detrended and normalized data                 #
#################################################################

stadial_mask, transition_ages = NGRIP_stadial_mask(d18o_binned_time)
interstadial_mask = ~stadial_mask
mask_name = '../detrended_data/stadial_mask.npy'
np.save(mask_name, stadial_mask)
fs = 16

fig = plt.figure(figsize=(14, 10))
width_ratios = [1, 0.1, 0.9] + [1] * 8
height_ratios = [1.3] * 10 + [1,6] + [1] * 10
gs = fig.add_gridspec(22, 11,
                      hspace=0.5,
                      left=0.05,
                      bottom=0.15,
                      top=0.90,
                      right=0.90,
                      width_ratios = width_ratios,
                      height_ratios = height_ratios)

ax3 = fig.add_subplot(gs[2:7, 1:-4])
ax4 = fig.add_subplot(gs[5:10, 1:-4])
ax_temp = fig.add_subplot(gs[0:4, 1:-4])
ax1 = fig.add_subplot(gs[12:18, 3:])
ax2 = fig.add_subplot(gs[16:22, 3:])
ax_hist1 = fig.add_subplot(gs[12:18, 0:2])
ax_hist2 = fig.add_subplot(gs[16:22, 0:2])

axbg0 = fig.add_subplot(gs[:10, 1:-4])
axbg = fig.add_subplot(gs[12:, 3:])

ax_detr1 = fig.add_subplot(gs[0:5, -3:])
ax_detr2 = fig.add_subplot(gs[6:11, -3:])

make_patch_spines_invisible(ax_temp)
make_patch_spines_invisible(ax3)
make_patch_spines_invisible(ax4)
make_patch_spines_invisible(ax1)
make_patch_spines_invisible(ax2)
make_patch_spines_invisible(axbg)
make_patch_spines_invisible(axbg0)
make_patch_spines_invisible(ax_hist1)
make_patch_spines_invisible(ax_hist2)

ax3.set_zorder(axbg0.get_zorder() + 3)
ax3.plot(d18o_120_time / 1000, d18o_120, color=d18o_color)
ax3.set_ylabel('$\delta^{18}$O$\;$[\u2030]')
ax3.annotate('(b)', (0.94, 0.82),
             xycoords='axes fraction', fontsize=fs)
ax3.set_xlim(123, 9)
ax3.xaxis.set_visible(False)
ax3.yaxis.set_ticks_position('right')
ax3.yaxis.set_label_position('right')
ax3.spines['right'].set_visible(True)


axbg0.set_xlim(123, 9)
#axbg0.axvspan(60,26, color = 'lightskyblue', alpha = 0.5, edgecolor = None)
axbg0.xaxis.set_visible(False)
axbg0.yaxis.set_visible(False)
#ax3.axvspan(60,26, color = 'lightskyblue', alpha = 0.5, edgecolor = None)
#ax4.axvspan(60,26, color = 'lightskyblue', alpha = 0.5, edgecolor = None)
axbg0.axvline(60, ls='--')
axbg0.axvline(26, ls='--')
#
ax4.set_zorder(axbg0.get_zorder() + 4)
ax4.plot(dust_120_time / 1000, dust_120, color=dust_color)
ax4.xaxis.set_visible(False)
ax4.set_ylabel(r'-ln(dust [ml$^{-1}$])')
ax4.annotate('(c)', (0.94, 0.82),
             xycoords='axes fraction', fontsize=fs)
ax4.set_xlim(123, 9)
ax4.spines['left'].set_visible(True)

ax_temp.plot(d18o_120_time/1000, temp_i, color = 'C1')
ax_temp.set_xlim(123, 9)
ax_temp.spines['left'].set_visible(True)
ax_temp.xaxis.set_visible(True)
ax_temp.set_ylabel(r'$\Delta T [^{\circ}]$')
ax_temp.xaxis.set_ticks_position('top')
ax_temp.xaxis.set_label_position('top')
ax_temp.spines['top'].set_visible(True)
ax_temp.set_xlabel('age [kyr b2k]', labelpad = 20)
ax_temp.annotate('(a)', (0.94, 0.82),
                  xycoords='axes fraction', fontsize=fs)


axbg.xaxis.set_visible(False)
axbg.yaxis.set_visible(False)
axbg.set_xlim(60, 26)
for i in range(0, len(transition_ages[:-1]), 2):
    axbg.axvspan(transition_ages[i]/1000,
                 transition_ages[i+1]/1000,
                 alpha=0.2,
                 edgecolor=None,
                 facecolor='slategray',
                 zorder=1)

#axbg.axvline(60, color = 'lightskyblue', lw =2)
#axbg.axvline(26, color = 'lightskyblue', lw = 2)

d18o_ticks = np.array([-2.5, 0, 2.5])
dust_ticks = np.array([-1.5, 0, 1.5])

ax1.set_zorder(axbg.get_zorder() + 1)
ax1.spines['left'].set_visible(True)
ax1.plot(d18o_binned_time / 1000,
         d18o_binned_detrended_norm,
         color=d18o_color, zorder=10)
ax1.set_xlim(60, 26)
ax1.set_ylabel(d18o_label)
ax1.set_yticks(d18o_ticks)
ax1.xaxis.set_visible(False)

ax2.set_zorder(axbg.get_zorder() + 1)
ax2.yaxis.set_ticks_position('right')
ax2.yaxis.set_label_position('right')
ax2.spines['right'].set_visible(True)
ax2.spines['bottom'].set_visible(True)
ax2.plot(dust_binned_time / 1000,
         dust_binned_detrended_norm,
         color=dust_color, zorder=10)
ax2.set_xlim(60, 26)
ax2.set_xlabel('age [kyr b2k]', labelpad = 15)
ax2.set_ylabel(dust_label)
ax2.set_yticks(dust_ticks)

ax1.annotate('(f)', (0.96, 0.85),
             xycoords='axes fraction', fontsize=fs)
ax2.annotate('(g)', (0.96, 0.85),
             xycoords='axes fraction', fontsize=fs)

xy0 = (60, ax4.get_ylim()[0])
xy1 = (60, ax1.get_ylim()[1])

con1 = ConnectionPatch(xy0, xy1, 'data', 'data',
                       ax4, ax1,
                       ls='solid',
                       zorder=1,
                       color='C0',
                       lw=1)

ax1.add_artist(con1)

xy0 = (26, ax4.get_ylim()[0])
xy1 = (26, ax1.get_ylim()[1])

con1 = ConnectionPatch(xy0, xy1, 'data', 'data',
                       ax4, ax1,
                       ls='solid',
                       zorder=-100,
                       color='C0',
                       lw=1)
ax1.add_artist(con1)

#################################################################
# add the histograms                                            #
#################################################################

ax_hist1.spines['right'].set_visible(True)

hist, edge = np.histogram(d18o_binned_detrended_norm,
                          bins=30, density=True)
x=(edge[1:] + edge[:-1])/2
ax_hist1.barh(x, height=np.mean(np.gradient(x))*0.8, width=hist,
          color=d18o_color, lw=1, edgecolor='k')
ax_hist1.invert_xaxis()
ax_hist1.xaxis.set_ticklabels([])
ax_hist1.yaxis.set_ticklabels([])
ax_hist1.set_yticks(d18o_ticks)
ax_hist1.yaxis.set_ticks_position('right')
ax_hist1.spines['top'].set_visible(True)
ax_hist1.set_xlabel('rel. frequency')
ax_hist1.xaxis.set_label_position('top')
ax_hist1.set_ylim(ax1.get_ylim())
ax_hist1.annotate('(h)', (-0.1, 0.80),
             xycoords='axes fraction', fontsize=fs)


ax_hist2.set_ylim(ax2.get_ylim())
ax_hist2.spines['right'].set_visible(True)
hist, edge = np.histogram(dust_binned_detrended_norm,
                          bins=30, density=True)
x=(edge[1:] + edge[:-1])/2
ax_hist2.barh(x, height=np.mean(np.gradient(x))*0.8, width=hist,
          color=dust_color, lw=1, edgecolor='k')
ax_hist2.invert_xaxis()
ax_hist2.yaxis.set_ticklabels([])
ax_hist2.xaxis.set_visible(False)
ax_hist2.set_yticks(dust_ticks)
ax_hist2.yaxis.set_ticks_position('right')


#############################################################
# create the scatter plot with the linear detrending        # 
#############################################################

make_patch_spines_invisible(ax_detr1)
make_patch_spines_invisible(ax_detr2)

ax_detr1.annotate('(d)',
             (0.86, 0.86),
             xycoords='axes fraction')


ax_detr1.yaxis.set_label_position('right')
ax_detr1.yaxis.set_ticks_position('right')

ax_detr1.scatter(temp_d18o[GI_d18o], d18o[GI_d18o],
                 label='GI data', s=10, facecolor = 'none',
                 edgecolor = d18o_color, linewidth = 0.3)
ax_detr1.scatter(temp_d18o[GS_d18o], d18o[GS_d18o],
                 label='GS data', s=10, edgecolor = 'slategray',
                 facecolor = 'none',
                 linewidth = 0.3,
                 alpha = 0.5)


ax_detr1.plot(np.sort(temp_d18o),
          np.sort(temp_d18o)*d18o_par[0] + d18o_par[1],
          color='k',
          lw=1.5,
          linestyle='dashed')
ax_detr1.plot(np.sort(temp_d18o),
          np.sort(temp_d18o)*d18o_par[0] + d18o_par[2],
          color='k',
          lw=1.5,
          linestyle='dashed')

ax_detr1.set_ylabel('$\delta^{18}$O [\u2030]', labelpad = 15)
ax_detr1.xaxis.set_ticks_position('top')
ax_detr1.xaxis.set_visible(False)
ax_detr1.spines['right'].set_visible(True)

#ax_detr1.set_xticks([-1,0,1,2])
# l2 = ax_detr1.legend(frameon=False,
#                 bbox_to_anchor=(-0.1, 1.2),
#                 loc='lower left',
#                 markerscale = 6)

ax_detr1.annotate(r'$ \delta^{18}\text{O} = %0.2f \Delta T + \begin{cases} %0.2f \quad \text{if GI} \\ %0.2f \quad \text{if GS} \end{cases}$' % (d18o_par[0], d18o_par[1], d18o_par[2]),
             xy=(-5.7, -50),
             xycoords='data',
             fontsize = 10)

legend_ax = ax_detr1.twiny()
make_patch_spines_invisible(legend_ax)
legend_ax.xaxis.set_visible(False)
legend_ax.yaxis.set_visible(False)
legend_ax.scatter([],[], edgecolor = d18o_color, facecolor = 'none', label = '$\delta^{18}$O--GI')
legend_ax.scatter([],[], edgecolor = dust_color, facecolor = 'none', label = 'dust--GI')
legend_ax.scatter([],[], edgecolor = 'slategray', facecolor = 'none', label = 'GS')
l1 = legend_ax.legend(frameon=False,
                      bbox_to_anchor=(-0.12, 1.1, 0.8, 0.3),
                      loc='lower left',
                      ncol=3, markerscale = 1.5,
                      columnspacing = 0.4,
                      handletextpad = 0.4)

#l1.legendHandles[0].set_linewidth(3)
           
#############################################################
# create the scatter plot with the dust linear detrending   # 
#############################################################

ax_detr2.annotate('(e)',
             (0.86, 0.86),
             xycoords='axes fraction')

ax_detr2.yaxis.set_label_position('right')
ax_detr2.yaxis.set_ticks_position('right')

ax_detr2.scatter(temp_dust[GI_dust], dust[GI_dust],
                 label='GI data', s=10, edgecolor=dust_color,
                 facecolor = 'none', linewidth = 0.3)
ax_detr2.scatter(temp_dust[GS_dust], dust[GS_dust],
                 label='GS data', s=10, facecolor='none',
                 edgecolor = 'slategray',
                 linewidth = 0.3,
                 alpha = .5)

ax_detr2.plot(np.sort(temp_dust),
          np.sort(temp_dust)*dust_par[0] + dust_par[1],
          color='k',
          label='$a = %0.2f, b = %0.2f$' % (dust_par[0], dust_par[1]),
          lw=1.5,
          linestyle='dashed')
ax_detr2.plot(np.sort(temp_dust),
          np.sort(temp_dust)*dust_par[0] + dust_par[2],
          color='k',
          label=(r'$a = %0.2f, b_{\text{GS}} = %0.2f, b_{\text{GI}} = %0.2f$'
                 % (dust_par[0], dust_par[1], dust_par[2])),
          lw=1.5,
          linestyle='dashed')

ax_detr2.set_ylabel(r'-ln(dust [ml$^{-1}$])', labelpad = 10)
ax_detr2.set_xlabel(r'$\Delta T$')
#ax_detr2.set_xticks([-1,0,1,2])

ax_detr2.annotate(r'dust$ = %0.2f \Delta T + \begin{cases} %0.2f \quad \text{if GI} \\ %0.2f \quad \text{if GS} \end{cases}$' % (dust_par[0], dust_par[1], dust_par[2]),
             xy=(-5.5, -14),
             xycoords='data',
             fontsize = 10)
ax_detr2.spines['bottom'].set_visible(True)
ax_detr2.spines['right'].set_visible(True)


#l4 = ax4.legend(frameon=False,
#                bbox_to_anchor=(0, 2.30),
#                loc='lower left',
#                ncol=2)
#l4.legendHandles[0].set_linewidth(3)

#fig.savefig('../figures/detrending_temp.pdf')


fig.savefig('../figures/fig01.pdf', transparent=True)
