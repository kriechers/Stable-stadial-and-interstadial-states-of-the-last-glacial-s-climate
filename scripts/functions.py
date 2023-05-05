import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import filtfilt, cheby1, lombscargle
from scipy.optimize import fmin


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def vmarker(x0, x1, ax0, ax1, **kwargs):
    xy0 = (x0, ax0.get_ylim()[0])
    xy1 = (x1, ax1.get_ylim()[1])
    ax0.axvline(x0, **kwargs)
    ax1.axvline(x1, **kwargs)
    con = ConnectionPatch(xy0, xy1, 'data', 'data',
                          ax0, ax1, **kwargs)
    ax0.add_artist(con)



def binning(t_ax, data, bins=None):
    '''
    INPUT
    -----
    data:= data of a time series
    t_ax:= time axis of a time series
    bins:= array like, equally spaced. 
        centers of bins will define new t_axis


    Output

    binned_data:= the i-th entry of binned_data is the 
        the mean of all data points which are located 
        between the i-th and i+1th elements of bins on the 
        t_ax. 
    binned_t_ax := center value of the bins
    '''

    if bins is None:
        res = np.max(np.diff(t_ax))
        bins = np.arange(t_ax[0]-res // 2,
                         t_ax[-1]+res,
                         res)

    binned_data = np.array([np.mean(data[(t_ax > bins[i]) &
                                         (t_ax < bins[i+1])])
                            for i in range(len(bins)-1)])

    binned_t_ax = bins[:-1]+0.5*(bins[1]-bins[0])

    idx = np.argwhere(np.isnan(binned_data))
    diff = np.diff(idx.flatten())
    mask = (np.append(diff, 0) == 1) + (np.append(0, diff) == 1)

    if any(np.diff(idx.flatten()) == 1):
        print('adjacent empty data')

    single_skip = idx[~mask]

    for i in single_skip:
        binned_data[i] = (binned_data[i-1] + binned_data[i+1]) / 2

    return binned_t_ax, np.array(binned_data)


def NGRIP_stadial_mask(age):
    '''
    this function takes an age axis as an input, that is compatibles 
    with the NGRIP GICC05modelext chronology in terms of covered time. 

    it returen a mask of the same length as the input age axis, which 
    is true, where the age of input corresponds a Greenland stadial. 
    '''

    # load dataset on GI onsets and GS onsets
    stratigraphic = pd.read_excel('../data/Rasmussen_et_al_2014'
                                  + '_QSR_Table_2.xlsx',
                                  header=None,
                                  skiprows=range(23),
                                  names=['event', 'age'],
                                  usecols='A,C')

    # create a mask from the above data which is True for all
    # GI events
    stadials = np.array(['GS' in x for x in stratigraphic['event']])

    # from that mask, a mask is derived that selects only transitions
    # between GI and GS from the stratigraphic data set (the set includes
    # minor GI events that follow major events => GI to GI to GS for example)
    # if transitions[i] is True, there is a transition between the i-th and the
    # i+1-th event from the stratigraphic data set. Since time goes in the
    # opposite direction than age, the age corresponding to the i-th event
    # is the correct age of the transition (that is the point in time where
    # the preceeding phase ended).

    transitions = np.append(np.array(stadials[:-1]) != np.array(stadials[1:]),
                            False)
    transition_ages = stratigraphic['age'][transitions].values

    max_age = np.max(age)
    min_age = np.min(age)

    start_idx = 0
    while transition_ages[start_idx] < min_age:
        start_idx += 1

    end_idx = len(transition_ages)-1
    while transition_ages[end_idx] > max_age:
        end_idx -= 1

    if stadials[start_idx]:
        GS = age < transition_ages[start_idx]

        for i in range(start_idx + 1, end_idx, 2):
            GS_mask = ((transition_ages[i] < age)
                       & (age < transition_ages[i+1]))
            GS = GS | GS_mask
    else:
        GS = np.full(len(age), False)
        for i in range(start_idx, end_idx, 2):
            GS_mask = ((transition_ages[i] < age)
                       & (age < transition_ages[i+1]))
            GS = GS | GS_mask

    return GS, transition_ages[start_idx: end_idx+1]


def cheby_lowpass(cutoff, fs, order, rp):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = cheby1(order, rp, normal_cutoff, btype='low', analog=False)
    return b, a


def cheby_lowpass_filter(x, cutoff, fs, order, rp):
    b, a = cheby_lowpass(cutoff, fs, order, rp)
    y = filtfilt(b, a, x)
    return y


def cheby_highpass(cutoff, fs, order, rp):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = cheby1(order, rp, normal_cutoff, btype='high', analog=False)
    return b, a


def cheby_highpass_filter(x, cutoff, fs, order, rp):
    b, a = cheby_highpass(cutoff, fs, order, rp)
    y = filtfilt(b, a, x)
    return y


def detrend_temperature(data, time, ax):
    #############################################################
    # import LR04 data and prepare the interpolated LR04 data   #
    # for detrending                                            # 
    #############################################################

    t_i = time[0]
    t_f = time[-1]

    temp_source = '../data/41586_2016_BFnature19798_MOESM245_ESM.xlsx'

    temperature = pd.read_excel(temp_source,
                                usecols=[0, 2],
                                skiprows=1,
                                names=['age', 'T_anom'])

    time2temp = interp1d(temperature['age']*1000, temperature['T_anom'])
    temp_i = time2temp(time)

    #temp_i = (temp_i - np.mean(temp_i))/np.std(temp_i)
    

    #############################################################
    # define masks for the GS and GI                            #
    #############################################################

    GS, transition_ages = NGRIP_stadial_mask(time)
    transition_ages = np.concatenate(([t_i], transition_ages, [t_f]))
    transition_idx = np.argwhere(np.diff(GS)).flatten()+1
    transition_idx = np.concatenate([[0],transition_idx, [len(time)]])
    GI = ~GS

    #############################################################
    # apply linear fit to the data                              #
    #############################################################

    GI_a, GI_b = np.polyfit(temp_i[GI], data[GI], 1)
    GS_a, GS_b = np.polyfit(temp_i[GS], data[GS], 1)

    p0 = ((GI_a + GS_a)/2, GI_b, GS_b)

    def rmse(p):
        temp = (np.where(GI,
                         data - p[1] - p[0] * temp_i,
                         data - p[2] - p[0] * temp_i) ** 2)
        temp = np.sqrt(np.sum(temp**2))
        return temp

    p_opt = fmin(lambda p: rmse(p), p0)

    opt_slope = p_opt[0]
    data_combined_detrended = data - (opt_slope* temp_i)

    ax.scatter(temp_i[GI], data[GI], color = 'C0')
    ax.plot(temp_i, temp_i * GI_a + GI_b, color = 'C4')
    ax.scatter(temp_i[GS], data[GS], color = 'C1')
    ax.plot(temp_i, temp_i * GS_a + GS_b, color = 'C3')
    ax.plot(temp_i, temp_i * opt_slope + (p_opt[1] + p_opt[2])/2)
    
    return data_combined_detrended, temp_i, GI, GS, p_opt
