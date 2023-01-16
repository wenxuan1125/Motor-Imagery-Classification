# http://www.bbci.de/competition/iv/desc_2b.pdf
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import string
import mne
import wandb
from torchsummary import summary
import read_my # read_my.py
import read_bci # read_bci.py
import cnn_stft # cnn_stft.py
import preprocess # preprocess.py
import plot_cf # plot_cf.py
import globalvar as gl # globalvar.py

#########################################################################
#                             Data Settings                             #
#########################################################################

gl._init()

dates = ['1006', '1013'] # for my experiment data
# option: '1006', '1013', '1027'

gl.set_value('augmentation_offset_start', -0.2) # 0.2 second before MI start
gl.set_value('augmentation_offset_end'  ,  0.1)    # 0.1 second after MI start

gl.set_value('baseline_offset_start', -0.5) # 0.5 second before MI start
gl.set_value('baseline_offset_end'  , -0.1)   # 0.1 second before MI start

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

#########################################################################
#                     Plotting Data Initialization                      #
#########################################################################

# for figure 1 - Confusion Matrix
cf_y_truth = []
cf_y_pred  = []

# for figure 2 - Accuracy vs Epoch
x_axis_epoch = []
y_axis_val_accuracy  = []
y_axis_train_accuracy = []

# for figure 3 - Loss vs Epoch
y_axis_val_loss  = []
y_axis_train_loss = []

#########################################################################
#                               Training                                #
#########################################################################
def train(args, model, device, train_loader, optimizer, epoch):
    global y_axis_train_accuracy, y_axis_train_loss

    model.train()
    total_correct = 0 # num of correct predictions in this epoch

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        data   = data.float()
        target = target.long()
        output = model(data)
        loss   = F.cross_entropy(output, target)

        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item() # num of correct predictions in this batch
        total_correct += correct

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}/{} {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                correct, len(target), 100. * correct / len(target)))

            if args.dry_run:
                break

    y_axis_train_accuracy.append(100. * total_correct / len(train_loader.dataset))
    y_axis_train_loss.append(loss.item())

#########################################################################
#                              Validating                               #
#########################################################################
def validation(model, device, val_loader):
    global y_axis_val_accuracy, cf_y_truth, cf_y_pred, y_axis_val_loss

    model.eval()
    val_loss = 0
    total_correct = 0 # num of correct predictions in this epoch

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            data   = data.float()
            target = target.long()
            output = model(data)
            val_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            
            for i in range(len(target)):
                cf_y_truth.append(target[i].cpu().item())
                cf_y_pred.append(pred[i][0].cpu().item())

    val_loss /= len(val_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, total_correct, len(val_loader.dataset),
        100. * total_correct / len(val_loader.dataset)))    
    
    y_axis_val_accuracy.append(100. * total_correct / len(val_loader.dataset))
    y_axis_val_loss.append(val_loss)

#########################################################################
#                             Main Function                             #
#########################################################################

def main():
    global x_axis_epoch, y_axis_val_accuracy, y_axis_train_accuracy, cf_y_truth, cf_y_pred, y_axis_val_loss, y_axis_train_loss
    
    ############################### Training Settings ###############################
    parser = argparse.ArgumentParser(description='BCI_offline')
    parser.add_argument('dataset', help='input dataset: bci/my')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--val-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for validation (default: 500)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='Loading the Pre-trained Model')  
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    train_kwargs = {'batch_size': args.batch_size}
    val_kwargs = {'batch_size': args.val_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 2,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)

    ############################### Loading Data ###############################
    if args.dataset == 'bci':
        # gl.set_value('is_scale_up', False)
        gl.set_value('is_filtering', False)
        gl.set_value('is_data_augmentation', False)
        gl.set_value('is_baseline_correction', True)
        gl.set_value('is_normalization', True)
        gl.set_value('is_magic', False)

        train_data, train_label, val_data, val_label = read_bci.read_bci_data()
        train_data  = torch.from_numpy(train_data)
        train_label = torch.from_numpy(train_label)
        val_data   = torch.from_numpy(val_data)
        val_label  = torch.from_numpy(val_label)
    elif args.dataset == 'my':
        # gl.set_value('is_scale_up', False)
        gl.set_value('is_filtering', True)
        gl.set_value('is_data_augmentation', True)
        gl.set_value('is_baseline_correction', True)
        gl.set_value('is_normalization', True)
        gl.set_value('is_magic', False)

        train_data, train_label, val_data, val_label = read_my.read_my_experiment(dates)
        train_data  = torch.from_numpy(train_data)
        train_label = torch.from_numpy(train_label)
        val_data   = torch.from_numpy(val_data)
        val_label  = torch.from_numpy(val_label)

    train_loader = torch.utils.data.DataLoader(TensorDataset(train_data, train_label),**train_kwargs)
    val_loader  = torch.utils.data.DataLoader(TensorDataset(val_data, val_label), **val_kwargs)

    ############################### Model Setup ###############################
    model = cnn_stft.CNN_STFT().to(device)
    summary(model, (3, 750))    # (3, 750) is input data size ( not including batch size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)

    # SGD + CyclicLR
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.003, step_size_up=10, mode='exp_range', gamma=0.99994)

    ###################### Loading And Saving Model Settings ######################
    if args.save_model:
        save_index = input("Save file's initial index: ")
        save_index = int(save_index)
        save_frequency = input("Save every ? epochs: ")
        save_frequency = int(save_frequency)
    
    # load pre-trained model
    if_train = 'y'
    if args.load_model:
        if_train = input('Train ?, y/n: ')
        file_dir = input("Load file's directory: ")
        file_index = input("Load file's index: ")

        # load model
        filename = './model_saved/' + file_dir + '/' + file_index + '.pt'
        model.load_state_dict(torch.load(filename))

    # only load the model for validation
    if args.load_model and if_train == 'n': 
        validation(model, device, val_loader) 

    # load the model for fine-tuning and validate / train new model
    else: 
        # train and validate for args.epochs times
        for current_epoch in range(1, args.epochs + 1):
            x_axis_epoch.append(current_epoch)
            train(args, model, device, train_loader, optimizer, current_epoch)
            validation(model, device, val_loader)

            # save model every save_frequency epochs
            if args.save_model:
                if current_epoch % save_frequency == 0: 
                    filename = './model_saved/' + args.dataset + '/' + str(save_index) + '.pt'
                    torch.save(model.state_dict(), filename)
                    save_index += 1
            
            # update learning rate
            # scheduler.step()

    ############################### Plotting ###############################
    plt.figure(1)
    plot_cf.plot_confusion_matrix(cf_y_truth, cf_y_pred, ['0', '1'], title='Confusion Matrix')

    if if_train == 'y':
        plt.figure(2)
        plt.plot(x_axis_epoch, y_axis_val_accuracy, label = 'Validation Set')
        plt.plot(x_axis_epoch, y_axis_train_accuracy, label = 'Training Set')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy(%)')
        plt.legend()

        plt.figure(3)
        plt.plot(x_axis_epoch, y_axis_val_loss, label = 'Validation Set')
        plt.plot(x_axis_epoch, y_axis_train_loss, label = 'Training Set')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

    plt.show()

if __name__ == '__main__':
    main()