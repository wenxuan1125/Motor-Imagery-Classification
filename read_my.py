# http://www.bbci.de/competition/iv/desc_2b.pdf
import numpy as np
import string
from sklearn.model_selection import train_test_split
import mne
import csv
import random
from sklearn import utils
import preprocess # preprocess.py
import globalvar as gl # globalvar.py

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

    train_data, train_label, val_data, val_label = [], [], [], []

    for date in dates: # read each experiment day's data
        for session_id in range(1, 4): # 3 session each experiment day

            data_filepath  = './Experiment data/' + date + '_subject1_session{}_data.edf'.format(session_id)
            label_filepath = './Experiment data/' + date + '_subject1_session{}_label.csv'.format(session_id)
            event_filepath = './Experiment data/' + date + '_subject1_session{}_event.csv'.format(session_id)
            
            temp_train_data, temp_train_label, temp_val_data, temp_val_label = read_session(data_filepath, label_filepath, event_filepath)
            
            if gl.get_value('is_magic'):
                if date == '1006':
                    temp_train_label = [0]*len(temp_train_label)
                    temp_val_label = [0]*len(temp_val_label)
                if date == '1013':
                    temp_train_label = [1]*len(temp_train_label)
                    temp_val_label = [1]*len(temp_val_label)

            train_data.extend(temp_train_data)
            train_label.extend(temp_train_label)
            val_data.extend(temp_val_data)
            val_label.extend(temp_val_label)
    
    # list to ndarray
    train_data  = np.squeeze(train_data) 
    train_label = np.squeeze(train_label)
    val_data   = np.squeeze(val_data)
    val_label  = np.squeeze(val_label)

    # preprocess #6: data augmentation - part 2
    if gl.get_value('is_data_augmentation'):
        train_data = preprocess.add_noise(train_data)

    # preprocess #7: normalization
    if gl.get_value('is_normalization'):

        train_data = preprocess.normalize(train_data)
        val_data = preprocess.normalize(val_data)

    # if gl.get_value('is_magic'):
    #     print(train_data[train_label==0, :, :].min())
    #     print(train_data[train_label==0, :, :].max())
    #     print(train_data[train_label==0, :, :].mean())
    #     print(train_data[train_label==0, :, :].std())
    #     print(train_data[train_label==1, :, :].min())
    #     print(train_data[train_label==1, :, :].max())
    #     print(train_data[train_label==1, :, :].mean())
    #     print(train_data[train_label==1, :, :].std())

    train_data, train_label = utils.shuffle(train_data, train_label)

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

    return train_data, train_label, val_data, val_label

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

    # split
    train_data_left, val_data_left, train_label_left, val_label_left     = train_test_split(trial_data_left, [0]*len(trial_data_left), test_size=gl.get_value('val_size'))
    train_data_right, val_data_right, train_label_right, val_label_right = train_test_split(trial_data_right, [1]*len(trial_data_right), test_size=gl.get_value('val_size'))

    train_data, train_label, val_data, val_label = [], [], [], []

    train_data.extend(train_data_left)
    train_data.extend(train_data_right)

    train_label.extend(train_label_left)
    train_label.extend(train_label_right)

    val_data.extend(val_data_left)
    val_data.extend(val_data_right)

    val_label.extend(val_label_left)
    val_label.extend(val_label_right)

    return train_data, train_label, val_data, val_label

############################### Read Data File ###############################
def extract_data(data_filename):

    raw = mne.io.read_raw_edf(data_filename, verbose='CRITICAL') 

    # preprocess #0: downsampling          
    raw.resample(sfreq=gl.get_value('sampling_rate'))   
    
    # preprocess #1: re-reference to average
    raw.c(ref_channels='average')

    raw = raw.drop_channels(gl.get_value('drop_channel_list'))

    # raw.get_data() is a ndarray, storing the sampled data
    data = raw.get_data()

    # preprocess #2: scale up from volt to microvolt
    # if gl.get_value('is_scale_up'):
        # data = data * 10e6

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
    pos = pos/(gl.get_value('original_sampling_rate') / gl.get_value('sampling_rate')) # because of downsampling
    pos = pos.astype('uint32')
    
    typ = np.squeeze(typ)
    typ = np.where(typ == 'boundary', 0, typ)   # if typ's element value == 'boundary', change this element value to 0, else remains its value
    typ = np.where(typ == 'S  1', 1, typ)       # if typ's element value == 'S  1', change this element value to 1, else remains its value
    typ = np.where(typ == 'S  2', 2, typ)       # if typ's element value == 'S  2', change this element value to 2, else remains its value
    typ = typ.astype('uint32')
    
    return np.array(list(zip(pos, typ)))