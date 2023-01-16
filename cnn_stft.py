import torch.nn as nn
import string
import torch
from torchaudio.transforms import Spectrogram
import matplotlib.pyplot as plt
import numpy as np

######################################################### Spectrogram #########################################################
# 
# Original:      kapre.time_frequency.Spectrogram(n_dft = 128, n_hop = 16, input_shape = input_shape, return_decibel_spectrogram = False, 
#                                                 power_spectrogram = 2.0, trainable_kernel = False, name = 'static_stft')
#                kapre.utils.Normalization2D(str_axis = 'freq')
# Default:       Not Found
# Documentation: Not Found
# 
# -----------------------------------------------------------------------------------------------------------------------------
# 
# Alternative:   torchaudio.transforms.Spectrogram(n_fft = 128, hop_length = 16, power = 2.0, normalized = True)
# Default:       torchaudio.transforms.Spectrogram(n_fft: int = 400, win_length: Optional[int] = None, hop_length: Optional[int] = None,
#                                                  pad: int = 0, window_fn: Callable[[...], torch.Tensor] = <built-in method hann_window of type object>,
#                                                  power: Optional[float] = 2.0, normalized: bool = False, wkwargs: Optional[dict] = None,
#                                                  center: bool = True, pad_mode: str = 'reflect', onesided: bool = True, return_complex: bool = True)
# Documentation: https://pytorch.org/audio/stable/transforms.html#torchaudio.transforms.Spectrogram
# 
########################################################### Conv2D ############################################################
# 
# Original:      keras.layers.Conv2D(filters = 24, kernel_size = (12, 12), strides = (1, 1), name = 'conv1', border_mode = 'same')
# Default:       keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding="valid", data_format=None, dilation_rate=(1, 1),
#                                    groups=1, activation=None, use_bias=True, kernel_initializer="glorot_uniform",
#                                    bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None
#                                    kernel_constraint=None, bias_constraint=None, **kwargs)
# Documentation: https://keras.io/api/layers/convolution_layers/convolution2d/
# 
# -----------------------------------------------------------------------------------------------------------------------------
# 
# Alternative:   torch.nn.Conv2d(3, 24, kernel_size=(12, 12), stride=(1, 1), bias=True, padding='same', padding_mode='zeros', dilation=(1,1))
# Default:       torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
#                                padding_mode='zeros', device=None, dtype=None)
# Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
# 
########################################################## BatchNorm ##########################################################
# 
# Original:      keras.layers.BatchNormalization(axis = 1)
# Default:       tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros",
#                                                   gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones",
#                                                   beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, **kwargs)
# Documentation: https://keras.io/api/layers/normalization_layers/batch_normalization/
# 
# -----------------------------------------------------------------------------------------------------------------------------
#
# Alternative:   torch.nn.BatchNorm2d(24, eps=0.001, momentum=0.99, affine=True, track_running_stats=True)
# Default:       torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
# Documentation: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
# 
######################################################### MaxPooling2D #########################################################
# 
# Original:      keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2,2), padding = 'valid', data_format = 'channels_last')
# Default:       keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None, **kwargs)
# Documentation: https://keras.io/api/layers/pooling_layers/max_pooling2d/
# 
# -----------------------------------------------------------------------------------------------------------------------------
#
# Alternative:   torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
# Default:       torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
# Documentation: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
# 
################################################################################################################################

class CNN_STFT(nn.Module):
    def __init__(self):
        super(CNN_STFT, self).__init__()

        # spectrogram creation using STFT
        self.stft = Spectrogram(n_fft=128, hop_length=16, power=2, normalized = True)                

        # Conv Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=(12, 12), stride=(1, 1), bias=True, padding='same', padding_mode='zeros', dilation=(1,1)),    
            nn.BatchNorm2d(24, eps=0.001, momentum=0.99, affine=True, track_running_stats=True),
            # 1.
            # keras.Conv2D -> channel_last(default), pytorch.Conv2D -> channel_first
            # keras.BatchNormalization(axis = 1): normalize axis 1
            # keras document mentions that when Conv2D layer with data_format="channels_first"(batch_size, channels, height, width), set axis=1
            # however, in default, data_format="channel_last"(batch_size, height, width, channels) in keras.Conv2D. axis is still set to 1 here -> a little strange(?)
            # also, in pytorch, no parameter can be set to change the normalized axis
            # 2. 
            # in pytorch, when affine=True, there are two parameter, gamma, beta which are initialized as uniform(0,1) and 0
            # in keras, affine is determined by two parameters, scale and center, in default both scale and center are set to True
            # but the gamma and beta are initialized as 1 and 0. gamma differs from pytorch, but we can't change the gamma initialization in pytorch

            # original: keras.layers.Activation('relu')
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0) # in pytorch, there are three parameters, dilation, return_indices and left_power which cannot be set in keras
        )

        # Conv Block 2
        self.conv2 = nn.Sequential(
            # original: keras.layers.Conv2D(filters = 48, kernel_size = (8, 8), name = 'conv2', border_mode = 'same')
            nn.Conv2d(24, 48, kernel_size=(8, 8), stride=(1, 1), bias=True, padding='same', padding_mode='zeros', dilation=(1,1)),
            nn.BatchNorm2d(48, eps=0.001, momentum=0.99, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        )

        # Conv Block 3
        self.conv3 = nn.Sequential(
            # original: keras.layers.Conv2D(filters = 96, kernel_size = (4, 4), name = 'conv3', border_mode = 'same')
            nn.Conv2d(48, 96, kernel_size=(4, 4), stride=(1, 1), bias=True, padding='same', padding_mode='zeros', dilation=(1,1)),
            nn.BatchNorm2d(96, eps=0.001, momentum=0.99, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            # original: keras.layers.Dropout(dropout=0.5)
            nn.Dropout(0.5)
        )
        
        # classificator
        # original keras model: Dense(2)  # two classes only
        # original keras model: Activation('softmax') -> implemented in cross_entropy
        self.fc4 = nn.Linear(in_features=3840, out_features=2, bias=True)

    def forward(self, x):
        x = self.stft(x)

        # print('/////////////////',x[0, :, :, :].cpu().numpy().shape,'/////////////////////') #(3, 65, 47)
        # x = x[0, :, :, :].cpu().numpy()
        # x = np.transpose(x, (1, 2, 0)) #(65, 47, 3)
        # x_max = x.max()
        # x = x/x_max
        # print('/////////////////',x.mean(),'/////////////////////')
        # print(x)
        # plt.imshow(x)
        # plt.figure()
        # plt.show()

        x = self.conv1(x) 
        x = self.conv2(x)
        x = self.conv3(x)
        
        # original keras model: Flatten()
        x = x.view(x.size(0), -1)
        # print(x.shape)
        output = self.fc4(x)

        return output