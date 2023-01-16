# http://www.bbci.de/competition/iv/desc_2b.pdf
import numpy as np
import string
import mne
import globalvar as gl # globalvar.py

############################### Average Rereference ###############################
def average_rereference(data):
    data_mean = data.mean()
    data = data - data_mean
    return data

############################### Filters ###############################
def filtering(data):

    data = mne.filter.notch_filter(data, Fs=gl.get_value('sampling_rate'), freqs=60, verbose='warning')
    data = mne.filter.filter_data(data, sfreq=gl.get_value('sampling_rate'), l_freq=0.5, h_freq=None, verbose='warning')
    data = mne.filter.filter_data(data, sfreq=gl.get_value('sampling_rate'), l_freq=2, h_freq=40, verbose='warning')
    return data

############################### Normalization ###############################
def normalize(data):

    for data_i in range(data.shape[0]):
        signal = data[data_i, :, :]
        temp = signal - signal.mean()
        temp = temp / signal.std()
        data[data_i, :, :] = temp

    return data

############################### Baseline Correction ###############################
def baseline_correction(baseline, data):

    baseline_mean = baseline.mean(axis=1)
    for i in range(data.shape[1]):
        data[:, i] = data[:, i] - baseline_mean
    return data

############################### Data Augmentation ###############################
def add_noise(data):

    num_of_sample_points = gl.get_value('sampling_rate') * gl.get_value('MI_period')
    for data_i in range(data.shape[0]):

        random_array = np.random.rand(gl.get_value('num_of_channel'), num_of_sample_points) # range [0, 1]
        # noise range is 1/100 of the data range
        # e.g. data range -> [-100, 900], noise range -> 10, noise -> [-5, 5]
        noise_range = (data[data_i, :, :].max() - data[data_i, :, :].min())/100
        noise = (random_array - 0.5) * noise_range
        data[data_i, :, :] = data[data_i, :, :] + noise

    return data