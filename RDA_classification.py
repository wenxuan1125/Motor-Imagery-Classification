import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import mne
import preprocess # preprocess.py
import cnn_stft # cnn_stft.py

def test(model, device, test_loader):

    model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            data   = data.float()
            target = target.long()
            output = model(data)            
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            print('=================================')
            print('Truth: ', target.cpu().item())
            print('Predict: ', pred.cpu().item())
            print('=================================')

            # target = target[0].cpu().item()
            # pred = pred[0][0].cpu().item()

            # print('=================================')
            # print('Truth: ' + str(target))
            # print('Predict: ' + str(pred))
            # print('=================================')

# def classification_job(data3s, test_label, test_kwargs, model, device):
def classification_job(data3s, label, baseline, delete_channel, resolutions): 
    
    # test_loader  = torch.utils.data.DataLoader(TensorDataset(data3s, test_label), **test_kwargs)
    # test(model, device, test_loader) 

    print('Classification job!')
    data3s = np.reshape(np.array(data3s), (32, 1500))
    baseline = np.reshape(np.array(baseline), (32, 200))
    resolutions = np.array(resolutions)
    # data*resolutions
    for i in range(data3s.shape[1]):
        data3s[:,i] = data3s[:,i]*resolutions
    for i in range(baseline.shape[1]):
        baseline[:,i] = baseline[:,i]*resolutions  

    # down sample
    data3s = mne.filter.resample(data3s, down=2, npad='auto')
    baseline = mne.filter.resample(baseline, down=2, npad='auto')

    # average rereference
    for i in range(data3s.shape[1]):
        average = data3s[:, i].mean()
        data3s[:, i] = data3s[:, i] - average
    for i in range(baseline.shape[1]):
        average = baseline[:, i].mean()
        baseline[:, i] = baseline[:, i] - average


    # drop channels
    data3s = np.delete(data3s, delete_channel, axis=0)
    baseline = np.delete(baseline, delete_channel, axis=0)

    # filtering
    data3s = mne.filter.filter_data(data3s, sfreq=250, l_freq=2, h_freq=40, verbose='warning', )
    baseline = mne.filter.filter_data(baseline, sfreq=250, l_freq=2, h_freq=40, verbose='warning')

    # baseline correction
    baseline_mean = baseline.mean(axis=1)
    for i in range(data3s.shape[1]):
        data3s[:, i] = data3s[:, i] - baseline_mean

    
    # normalization
    temp = data3s - data3s.mean()
    temp = temp / data3s.std()
    data3s = temp

    print(data3s.shape)
    print(label)


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    test_kwargs = {'batch_size': 1}

    if use_cuda:
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True}
        test_kwargs.update(cuda_kwargs)

    # load model
    model = cnn_stft.CNN_STFT().to(device)
    filename = './model_saved/' + 'my' + '/' + '1' + '.pt'
    model.load_state_dict(torch.load(filename))

    # data loader
    test_data   = torch.from_numpy(np.array([data3s]))
    test_label  = torch.from_numpy(np.array([label]))
    test_loader  = torch.utils.data.DataLoader(TensorDataset(test_data, test_label), **test_kwargs)

    test(model, device, test_loader)


    