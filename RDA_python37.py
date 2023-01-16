# needs socket and struct library
import socket
import struct 
import sys
import threading
import torch
import cnn_stft # cnn_stft.py
import preprocess # preprocess.py
import plot_cf # plot_cf.py
import globalvar as gl # globalvar.py
import RDA_classification # RDA_classification.py

# Marker class for storing marker information
class Marker:
    def __init__(self):
        self.position = 0
        self.points = 0
        self.channel = -1
        self.type = ""
        self.description = ""

# Helper function for receiving whole message
def RecvData(socket, requestedSize):
    returnStream = b''
    while len(returnStream) < requestedSize:
        databytes = socket.recv(requestedSize - len(returnStream))
        if databytes == '':
            raise RuntimeError("connection broken")
        returnStream += databytes
 
    return returnStream 

# Helper function for splitting a raw array of
# zero terminated strings (C) into an array of python strings
def SplitString(raw):
    stringlist = []
    decode_raw = raw.decode()
    s = ""
    for i in range(len(decode_raw)):
        if decode_raw[i] != '\x00':        # '\x00' is null
            s = s + decode_raw[i]
        else:
            stringlist.append(s)
            s = ""

    return stringlist

# Helper function for extracting eeg properties from a raw data array
# read from tcpip socket
def GetProperties(rawdata):

    # Extract numerical data
    (channelCount, samplingInterval) = struct.unpack('<Ld', rawdata[:12])   # decoding rawdata[:12] to one unsigned long integer and one double
                                                                            # <: little-endian, L: unsigned long (4 bytes), d: double (8 bytes)
                                                                            
    # Extract resolutions
    resolutions = []
    for c in range(channelCount):
        index = 12 + c * 8
        restuple = struct.unpack('<d', rawdata[index:index+8])    # decoding one double (8 bytes)
        resolutions.append(restuple[0])

    # Extract channel names
    channelNames = SplitString(rawdata[12 + 8 * channelCount:])

    return (channelCount, samplingInterval, resolutions, channelNames)

# Helper function for extracting eeg and marker data from a raw data array
# read from tcpip socket       
def GetData(rawdata, channelCount):

    # Extract numerical data
    (block, points, markerCount) = struct.unpack('<LLL', rawdata[:12])

    # print('data points: ', points,  'data block: ', block)

    # Extract eeg data as array of floats
    data = []
    for i in range(points * channelCount):
        index = 12 + 4 * i
        value = struct.unpack('<f', rawdata[index:index+4])
        data.append(value[0])

    # Extract markers
    markers = []
    index = 12 + 4 * points * channelCount
    for m in range(markerCount):
        markersize = struct.unpack('<L', rawdata[index:index+4])

        ma = Marker()
        (ma.position, ma.points, ma.channel) = struct.unpack('<LLl', rawdata[index+4:index+16])
        typedesc = SplitString(rawdata[index+16:index+markersize[0]])
        ma.type = typedesc[0]
        ma.description = typedesc[1]

        markers.append(ma)
        index = index + markersize[0]

    return (block, points, markerCount, data, markers)

# # Create model
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
# model = cnn_stft.CNN_STFT().to(device)
# test_kwargs = {'batch_size': 200}
# if use_cuda:
#         cuda_kwargs = {'num_workers': 2,
#                        'pin_memory': True,
#                        'shuffle': True}
#         test_kwargs.update(cuda_kwargs)

# file_index = input("Load file's index: ")
# # load model
# filename = './model_saved/my/' + file_index + '.pt'
# model.load_state_dict(torch.load(filename))

# Create a tcpip socket
con = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
con.connect(("140.113.203.42", 51244))
print('CONECTION SUCCESS!!!')

# Flag for main loop
finish = False

# data buffer for calculation, empty in beginning
data3s = []
# baseline buffer
baseline = []
# extend data
dataExtend = False
baseExtend = False

# block counter to check overflows of tcpip buffer
lastBlock = -1

#### Main Loop ####
while not finish:

    # Get message header as raw array of chars
    rawhdr = RecvData(con, 24)                                                  # the first 24 bits in the packect are header

    # Split array into usefull information id1 to id4 are constants
    (id1, id2, id3, id4, msgsize, msgtype) = struct.unpack('<llllLL', rawhdr)   # header records id1, id2, id3, id4, msgsize, msgtype
                                                                                # packet size = msgsize 
    # print('size: ', msgsize, 'type: ', msgtype)
    # Get data part of message, which is of variable size                       
    rawdata = RecvData(con, msgsize - 24)                                       # packet size - header size(24) = data size

    # Perform action dependend on the message type
    if msgtype == 1:
        # Start message, extract eeg properties and display them
        (channelCount, samplingInterval, resolutions, channelNames) = GetProperties(rawdata)
        # reset block counter
        lastBlock = -1

        print("Start")
        print("Number of channels: " + str(channelCount))
        print("Sampling interval: " + str(samplingInterval))
        print("Resolutions: " + str(resolutions))
        print("Channel Names: " + str(channelNames))

        delete_channel = []
        for i, channel in enumerate(channelNames):
            if channel not in ['C3','Cz', 'C4']:
                delete_channel.append(i)

    elif msgtype == 4:
        # Data message, extract data and markers
        (block, points, markerCount, data, markers) = GetData(rawdata, channelCount)    # points = 10, channels = 32, 

        # Check for overflow
        if lastBlock != -1 and block > lastBlock + 1:
            print("*** Overflow with " + str(block - lastBlock) + " datablocks ***")
        lastBlock = block

        # Print markers, if there are some in actual block
        if markerCount > 0:
            for m in range(markerCount):
                # print("Marker " + markers[m].description + " of type " + markers[m].type)      

                # motor imagery trigger - left
                if markers[m].description == 'S  1':
                    dataExtend = True
                    data3s.extend(data[markers[m].position*32:])
                    label = 0   # left hand
                # motor imagery trigger - right
                elif markers[m].description == 'S  2':
                    dataExtend = True
                    data3s.extend(data[markers[m].position*32:])
                    label = 1   # right hand
                # motor imagery trigger - end
                elif markers[m].description == 'S  3':
                    dataExtend = False
                    data3s.extend(data[:markers[m].position*32])
                    data3s = data3s[:32*3*500]
                    baseline = baseline[:int(32*0.4*500)]

                    t = threading.Thread(target=RDA_classification.classification_job, args=(data3s, label, baseline, delete_channel, resolutions))
                    t.start()   #start the thread
                    data3s = []
                    baseline = []

                # baseline trigger - start
                elif  markers[m].description == 'S  4':
                    baseExtend = True
                    baseline.extend(data[markers[m].position*32:])
                # baseline trigger - end
                elif  markers[m].description == 'S  5':
                    baseExtend = False
                    baseline.extend(data[:markers[m].position*32])

        if dataExtend:
            data3s.extend(data)
        if baseExtend:
            baseline.extend(data)

    elif msgtype == 3:
        # Stop message, terminate program
        print("Stop")
        finish = True

con.close()
