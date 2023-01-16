# http://www.bbci.de/competition/iv/desc_2b.pdf
import numpy as np
import string
from sklearn.model_selection import train_test_split
import mne
import random
import preprocess # preprocess.py
import globalvar as gl # globalvar.py

############################# Preprocess Order ##############################
#   0. Re-reference to Average                                              #
#   1. Scale Up                                                             #
#   2. Filtering                                                            #
#   3. Data Augmentation - Part 1(Random Start)                             #
#   4. Baseline Correction                                                  #
#   5. Data Augmentation - Part 2(Add Noise)                                #
#   6. Normalization                                                        #
#############################################################################

############################### Read All Data ###############################
def read_bci_data():

    train_data, train_label, val_data, val_label = [], [], [], []

    for i in range(1, 10):
        for j in range(1, 4):

            subject = 'B0' + str(i) + '0' + str(j) + 'T'
            temp_train_data, temp_train_label, temp_val_data, temp_val_label = read_bci_subject(subject)

            train_data.extend(temp_train_data)
            train_label.extend(temp_train_label)
            val_data.extend(temp_val_data)
            val_label.extend(temp_val_label)

    train_data  = np.squeeze(train_data) # list to ndarray
    train_label = np.squeeze(train_label)
    val_data   = np.squeeze(val_data)
    val_label  = np.squeeze(val_label)

    # drop EOG channels, only keep C3 Cz C4
    train_data = np.delete(train_data, [3, 4, 5], 1)
    val_data  = np.delete(val_data, [3, 4, 5], 1)

    # preprocess #5: data augmentation - part 2
    if gl.get_value('is_data_augmentation'):
        train_data = preprocess.add_noise(train_data)

    # preprocess #6: normalization
    if gl.get_value('is_normalization'):
        train_data = preprocess.normalize(train_data)
        val_data = preprocess.normalize(val_data)

    print('================================================================')
    print('Preprocessing Order       ')
    print('================================================================')
    print('Down Sampling')
    print('Re-reference to Average')
    # if gl.get_value('is_scale_up'): print('Scale Up')
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

############################### Read One Subject ###############################
def read_bci_subject(name):

    filepath = './BCICIV_2b_gdf/' + name + '.gdf'
    raw = mne.io.read_raw_gdf(filepath, preload=True, verbose='CRITICAL')

    # preprocess #0: Re-reference to Average
    raw.set_eeg_reference(ref_channels='average')

    # using mne.events_from_annotations(raw) to get the event information
    # events is a ndarray, whose shape = (639, 3), containing the corresponding data number[0] and event type[2]
    # event_ids is a dictionary, containing the corresponding index (from 1 to 10) for the event type
    # event_ids = {'1023': 1, '1077': 2, '1078': 3, '1079': 4, '1081': 5, '276': 6, '277': 7, '32766': 8, '768': 9, '769': 10, '770': 11}
    # 768, id = 9: start of a trial
    # 769, id = 10: class 1(left hand MI)
    # 770, id = 11: class 2(right hand MI)
    # 1023, id = 1, rejected trial
    events, event_ids = mne.events_from_annotations(raw, verbose='CRITICAL')

    # only trial onset events
    left_hand_event     = event_ids['769']
    right_hand_event    = event_ids['770']
    trial_start_trigger = event_ids['768']

    trial_events_mask = [ev_code in [left_hand_event, right_hand_event] for ev_code in events[:,2]] # mark the usable event 
    trial_events = events[trial_events_mask]  
    
    trial_starts_mask = [ev_code in [trial_start_trigger] for ev_code in events[:,2]] # mark the trial_start_trigger event   
    trial_starts = events[trial_starts_mask]  
    starts = trial_starts[:, 0] # only the latency column

    # raw.get_data() is a ndarray, storing the sampled data
    data = raw.get_data()

    # preprocess #1: scale up from volt to microvolt
    # if gl.get_value('is_scale_up'):
        # data = data * 10e6

    # preprocess #2: filter the data
    if gl.get_value('is_filtering'):
        data = preprocess.filtering(data)

    train_label = trial_events[:, 2] - left_hand_event # label 0 -> left hand MI; label 1 -> right hand MI
    assert (len(train_label) == len(starts))
    
    train_data = []

    num_of_trial = np.array(train_label).shape[0]
    for i in range(num_of_trial):

        # preprocess #3: data augmentation - part 1
        if gl.get_value('is_data_augmentation'):
            start_offset = random.randint(gl.get_value('augmentation_offset_start') * gl.get_value('sampling_rate'), 
                                          gl.get_value('augmentation_offset_end')   * gl.get_value('sampling_rate'))
        else:
            start_offset = 0
        
        # motor imagery range
        MI_start = int(start_offset + starts[i] + gl.get_value('MI_start_time') * gl.get_value('sampling_rate'))
        MI_end   = int(start_offset + starts[i] + (gl.get_value('MI_start_time') + gl.get_value('MI_period')) * gl.get_value('sampling_rate'))
        
        # preprocess #4: baseline correction
        if gl.get_value('is_baseline_correction'):
            baseline_start = int(MI_start + gl.get_value('baseline_offset_start') * gl.get_value('sampling_rate'))
            baseline_end   = int(MI_start + gl.get_value('baseline_offset_end')   * gl.get_value('sampling_rate'))
            
            one_trial_data = preprocess.baseline_correction(data[:, baseline_start:baseline_end], data[:, MI_start:MI_end])
        else:
            one_trial_data = data[:, MI_start:MI_end]
        
        # append data
        train_data.append(one_trial_data)
        
    # split
    train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=gl.get_value('val_size'))

    return train_data, train_label, val_data, val_label
