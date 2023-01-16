# http://www.bbci.de/competition/iv/desc_2b.pdf
import numpy as np
import string
import mne
import csv
import random
from sklearn import utils
import matplotlib.pyplot as plt
import preprocess # preprocess.py
import globalvar as gl # globalvar.py

#########################################################################
#                             Data Settings                             #
#########################################################################

gl._init()

dates = ['1006'] # for my experiment data
# option: '1006', '1013', '1027'

gl.set_value('is_magic', True)
gl.set_value('is_scale_up', True)
gl.set_value('is_filtering', True)


gl.set_value('is_data_augmentation', True)
gl.set_value('augmentation_offset_start', -0.2) # 0.2 second before MI start
gl.set_value('augmentation_offset_end', 0.1)    # 0.1 second after MI start


gl.set_value('is_baseline_correction', True)
gl.set_value('baseline_offset_start', -0.5) # 0.5 second before MI start
gl.set_value('baseline_offset_end', -0.1)   # 0.1 second before MI start

gl.set_value('is_normalization', True)

gl.set_value('val_size', 0.1)  # 0.1 for validation and 0.9 for training

gl.set_value('keep_3_channel', ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'T7', 'TP9', 
                  'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 
                  'CP2', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'EOG', 'Status'])
gl.set_value('keep_7_channel', ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'T7', 'TP9', 
                  'CP5', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 
                  'T8', 'FT10', 'FC6', 'F4', 'F8', 'EOG', 'Status'])
gl.set_value('drop_channel_list', gl.get_value('keep_3_channel'))

#########################################################################
#                               Data Info                               #
#########################################################################

gl.set_value('num_of_channel', 33 - len(gl.get_value('drop_channel_list'))) # 31 EEG + 1 EOG + 1 Status = 33 channel
gl.set_value('original_sampling_rate', 500)     # 500 Hz
gl.set_value('sampling_rate', 250)              # 250 Hz
gl.set_value('MI_start_time', 4)                # MI starts at 4th second in each trial
gl.set_value('MI_period', 3)                    # MI last for 3 second

############################# Preprocess Order ##############################
#   0. Down Sampling                                                        #
#   1. Re-reference to Average                                              #
#   2. Scale Up                                                             #
#   3. Filtering                                                            #
#   4. Data Augmentation - Part 1(Random Start)                             #
#   5. Baseline Correction                                                  #
#   6. Data Augmentation - Part 2(Add Noise)                                #
#   7. Normalization                                                        #
#############################################################################

############################### Read All Data ###############################
def read_my_experiment(dates):

    # topomap settings
    channel_location = np.array([[-0.070711,0],[0,0],[0.070711, 0]])
    plt.figure(figsize=(10, 15))

    for i, date in enumerate(dates): # read each experiment day's data

        train_data_left, train_data_right = [], []

        for session_id in range(1, 4): # 3 session each experiment day

            data_filepath  = './Experiment data/' + date + '_subject1_session{}_data.edf'.format(session_id)
            label_filepath = './Experiment data/' + date + '_subject1_session{}_label.csv'.format(session_id)
            event_filepath = './Experiment data/' + date + '_subject1_session{}_event.csv'.format(session_id)
            
            temp_train_data_left, temp_train_data_right = read_session(data_filepath, label_filepath, event_filepath)

            train_data_left.extend(temp_train_data_left)
            train_data_right.extend(temp_train_data_right)
    
        # list to ndarray
        train_data_left  = np.squeeze(train_data_left) 
        train_data_right = np.squeeze(train_data_right)

        # preprocess #6: data augmentation - part 2
        if gl.get_value('is_data_augmentation'):
            preprocess.add_noise(train_data_left)
            preprocess.add_noise(train_data_right)

        # preprocess #7: normalization
        if gl.get_value('is_normalization'):
            preprocess.normalize(train_data_left)
            preprocess.normalize(train_data_right)
        
        # calculate power mean
        power_mean_left = power_mean(train_data_left)
        power_mean_right = power_mean(train_data_right)

        # plot topomap
        plt.subplot(320 + i * 2 + 1)

        vmin = min(np.min(power_mean_left), np.min(power_mean_right))
        vmax = min(np.max(power_mean_left), np.max(power_mean_right))

        im1, cm1 = mne.viz.plot_topomap(power_mean_left, channel_location, vmin=vmin, 
                                        vmax=vmax, show=False, names=['C3', 'Cz', 'C4'], show_names=True)
        plt.title('{date} WenHsin\'s Left Hand Imagery'.format(date=date))
        plt.colorbar(im1, shrink=0.7)

        plt.subplot(320 + i * 2 + 2)
        im2, cm2 = mne.viz.plot_topomap(power_mean_right, channel_location, vmin=vmin,
                                        vmax=vmax, show=False, names=['C3', 'Cz', 'C4'], show_names=True)
        plt.title('{date} WenHsin\'s Right Hand Imagery'.format(date=date))
        plt.colorbar(im2, shrink=0.7)

    print('================================================================')
    print('Preprocessing Order       ')
    print('================================================================')
    print('Down Sampling')
    print('Re-reference to Average')
    if gl.get_value('is_scale_up'): print('Scale Up')
    if gl.get_value('is_filtering'): print('Filtering')
    if gl.get_value('is_data_augmentation'): print('Data Augmentation')
    if gl.get_value('is_baseline_correction'): print('Baseline Correction')
    if gl.get_value('is_normalization'): print('Normalization')
    if gl.get_value('is_magic'): print('**Magic**')
    print('================================================================')

    print('****************************************************************')
    print('reading data completed!')
    print('****************************************************************')

    plt.show() 

    return

############################### Read One Session ###############################
def read_session(data_filepath, label_filepath, event_filepath):

    data   = extract_data(data_filepath)
    labels = extract_labels(label_filepath)
    events = extract_events(event_filepath)

    MI_start_trigger = 1 # MI_end_trigger = 2
    MI_starts_mask   = [ev_code in [MI_start_trigger] for ev_code in events[:, 1]]  # mark the MI_start_trigger event                                              
    MI_starts        = events[MI_starts_mask]
    starts           = MI_starts[:, 0]  # only the latency column
    assert (len(labels) == len(starts))

    trial_data_left, trial_data_right = [], []

    num_of_trial = labels.shape[0]
    for i in range(num_of_trial):

        # preprocess #4: data augmentation - part 1
        if gl.get_value('is_data_augmentation'):
            start_offset = random.randint(gl.get_value('augmentation_offset_start') * gl.get_value('sampling_rate'), 
                                          gl.get_value('augmentation_offset_end')   * gl.get_value('sampling_rate'))
        else:
            start_offset = 0

        # motor imagery range
        MI_start = int(start_offset + starts[i])
        MI_end   = int(start_offset + starts[i] + gl.get_value('MI_period') * gl.get_value('sampling_rate'))

        # preprocess #5: baseline correction
        if gl.get_value('is_baseline_correction'):
            baseline_start = int(MI_start + gl.get_value('baseline_offset_start') * gl.get_value('sampling_rate'))
            baseline_end   = int(MI_start + gl.get_value('baseline_offset_end')   * gl.get_value('sampling_rate'))
            
            one_trial_data = preprocess.baseline_correction(data[:, baseline_start:baseline_end], data[:, MI_start:MI_end])
        else:
            one_trial_data = data[:, MI_start:MI_end]

        # append data
        if labels[i] == 0:
            trial_data_left.append(one_trial_data)
        elif labels[i] == 1:
            trial_data_right.append(one_trial_data)

    return trial_data_left, trial_data_right

############################### Read Data File ###############################
def extract_data(data_filename):

    raw = mne.io.read_raw_edf(data_filename) 

    # preprocess #0: downsampling          
    raw.resample(sfreq=gl.get_value('sampling_rate'))   
    raw = raw.drop_channels(gl.get_value('drop_channel_list')) 

    # preprocess #1: re-reference to average
    raw.set_eeg_reference(ref_channels='average')

    # plot continuous eeg data
    # raw.plot()
    # plt.show()

    # raw.get_data() is a ndarray, storing the sampled data
    data = raw.get_data()

    # preprocess #2: scale up from volt to microvolt
    if gl.get_value('is_scale_up'):
        data = data * 10e6

    # preprocess #3: filter the data
    if gl.get_value('is_filtering'):
        preprocess.filtering(data)

    return data

############################### Read Label File ###############################
def extract_labels(label_filepath):

    with open(label_filepath) as csvFile:
        csvReader = csv.reader(csvFile)
        data = list(csvReader)

    label = [data[i][1] for i in range(1, len(data))]
    label = np.squeeze(label)
    label = label.astype('uint8')

    # label 1, 3 -> left hand MI; label 2, 4 -> right hand MI
    label = np.where(label == 3, 1, label)  # if label's element value == 3, change this element value to 1, else remains its value
    label = np.where(label == 4, 2, label)  # if label's element value == 4, change this element value to 2, else remains its value
    
    # label 0 -> left hand MI; label 1 -> right hand MI
    label = label - 1
    
    return label

############################### Read Event File ###############################
def extract_events(event_filepath):

    with open(event_filepath) as csvFile:
        csvReader = csv.reader(csvFile)
        data = list(csvReader)

    pos = [data[i][0] for i in range(1, len(data))] # latency
    typ = [data[i][3] for i in range(1, len(data))] # event type

    pos = np.squeeze(pos)
    pos = pos.astype('uint32')
    pos = pos/(gl.get_value('original_sampling_rate')/gl.get_value('sampling_rate')) # because of downsampling
    pos = pos.astype('uint32')
    
    typ = np.squeeze(typ)
    typ = np.where(typ == 'boundary', 0, typ)   # if typ's element value == 'boundary', change this element value to 0, else remains its value
    typ = np.where(typ == 'S  1', 1, typ)       # if typ's element value == 'S  1', change this element value to 1, else remains its value
    typ = np.where(typ == 'S  2', 2, typ)       # if typ's element value == 'S  2', change this element value to 2, else remains its value
    typ = typ.astype('uint32')
    
    return np.array(list(zip(pos, typ)))

############################# Calculate Power Mean #############################
def power_mean(data):
    power_sum = np.zeros(3)

    # data shape: (trails, channels, sample points) = (120, 3, 750)
    cnt = np.array(data).shape[0] * np.array(data).shape[2]

    for i in range(np.array(data).shape[0]):
        point_power = np.array(data)[i, 0:3]*np.array(data)[i, 0:3]
        power_sum += np.sum(point_power, axis=1)

    power_mean = pow(power_sum/cnt, 1.0/2)
    
    return power_mean

if __name__ == '__main__':

    dates = ['1006', '1013', '1027']
    read_my_experiment(dates)