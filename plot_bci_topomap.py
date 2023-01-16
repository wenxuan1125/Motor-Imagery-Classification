# http://www.bbci.de/competition/iv/desc_2b.pdf
import numpy as np
import string
import mne
import random
import matplotlib.pyplot as plt
import preprocess # preprocess.py
import globalvar as gl # globalvar.py

#########################################################################
#                             Data Settings                             #
#########################################################################

gl._init()

dates = ['1006'] # for my experiment data
# option: '1006', '1013', '1027'

gl.set_value('is_magic', False)
gl.set_value('is_scale_up', False)
gl.set_value('is_filtering', False)

gl.set_value('is_data_augmentation', False)
gl.set_value('augmentation_offset_start', -0.2) # 0.2 second before MI start
gl.set_value('augmentation_offset_end', 0.1)    # 0.1 second after MI start


gl.set_value('is_baseline_correction', False)
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

    # topomap settings
    channel_location = np.array([[-0.070711,0],[0,0],[0.070711, 0]])
    plt.figure(figsize=(20, 20))

    for i in range(1, 10):

        train_data_left, train_data_right = [], []

        for j in range(1, 4):

            subject = 'B0' + str(i) + '0' + str(j) + 'T'
            temp_train_data_left, temp_train_data_right = read_bci_subject(subject)
            
            train_data_left.extend(temp_train_data_left)
            train_data_right.extend(temp_train_data_right)

        # list to ndarray
        train_data_left  = np.squeeze(train_data_left) 
        train_data_right = np.squeeze(train_data_right)

        # preprocess #5: data augmentation - part 2
        if gl.get_value('is_data_augmentation'):
            preprocess.add_noise(train_data_left)
            preprocess.add_noise(train_data_right)

        # preprocess #6: normalization
        if gl.get_value('is_normalization'):
            preprocess.normalize(train_data_left)
            preprocess.normalize(train_data_right)

        # calculate power mean
        power_mean_left = power_mean(train_data_left)
        power_mean_right = power_mean(train_data_right)

        # plot topomap
        plt.subplot(5, 4,(i-1)*2+1)

        vmin = min(np.min(power_mean_left), np.min(power_mean_right))
        vmax = min(np.max(power_mean_left), np.max(power_mean_right))

        im1, cm1 = mne.viz.plot_topomap(power_mean_left, channel_location, vmin=vmin, 
                                        vmax=vmax, show=False, names=['C3', 'Cz', 'C4'], show_names=True)
        plt.title('Subject{id}\'s Left Hand Imagery'.format(id=str(i)), fontsize=9)
        plt.colorbar(im1, shrink=0.7)

        plt.subplot(5, 4,(i-1)*2+2)
        im2, cm2 = mne.viz.plot_topomap(power_mean_right, channel_location, vmin=vmin,
                                        vmax=vmax, show=False, names=['C3', 'Cz', 'C4'], show_names=True)
        plt.title('Subject{id}\'s Right Hand Imagery'.format(id=str(i)), fontsize=9)
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

############################### Read One Subject ###############################
def read_bci_subject(name):

    filepath = './BCICIV_2b_gdf/' + name + '.gdf'
    raw = mne.io.read_raw_gdf(filepath) # raw.get_data() is a ndarray, storing the sampled data
                                        # raw.get_data().shape = (25, 673328)
                                        # using mne.events_from_annotations(raw) to get the event information
                                        # events is a ndarray, whose shape = (639, 3), containing the corresponding data number[0] and event type[2]
                                        # event_ids is a dictionary, containing the corresponding index (from 1 to 10) for the event type
                                        # event_ids = {'1023': 1, '1077': 2, '1078': 3, '1079': 4, '1081': 5, '276': 6, '277': 7, '32766': 8, '768': 9, '769': 10, '770': 11}
                                        # 768, id = 9: start of a trial
                                        # 769, id = 10: class 1(left hand MI)
                                        # 770, id = 11: class 2(right hand MI)
                                        # 1023, id = 1, rejected trial

    # preprocess #0: Re-reference to Average
    raw.set_eeg_reference(ref_channels='average')

    events, event_ids = mne.events_from_annotations(raw)
    raw = raw.drop_channels(['EOG:ch01', 'EOG:ch02', 'EOG:ch03'])

    # plot continuous eeg data
    # raw.plot()
    # plt.show()

    # only trial onset events
    left_hand_event     = event_ids['769']
    right_hand_event    = event_ids['770']
    trial_start_trigger = event_ids['768']

    trial_events_mask = [ev_code in [left_hand_event, right_hand_event] for ev_code in events[:,2]] # mark the usable event 
    trial_events = events[trial_events_mask]  
    
    trial_starts_mask = [ev_code in [trial_start_trigger] for ev_code in events[:,2]] # mark the trial_start_trigger event   
    trial_starts = events[trial_starts_mask]  
    starts = trial_starts[:, 0] # only the latency column

    data = raw.get_data()

    # preprocess #1: scale up from volt to microvolt
    if gl.get_value('is_scale_up'):
        data = data * 10e6

    # preprocess #2: filter the data
    if gl.get_value('is_filtering'):
        preprocess.filtering(data)

    train_label = trial_events[:, 2] - left_hand_event # label 0 -> left hand MI; label 1 -> right hand MI
    assert (len(train_label) == len(starts))
    
    train_data_left, train_data_right = [], []

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

        # split into 2 class
        if train_label[i] == 0:
            train_data_left.append(one_trial_data)
        elif train_label[i] == 1:
            train_data_right.append(one_trial_data)
        
    return train_data_left, train_data_right

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
    read_bci_data()