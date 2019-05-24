
import logging
import os
from collections import OrderedDict
import math
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.modules.utils import _triple
try:
    from . import initialize
except:
    import initialize
#from torchsummary import summary

#################################################R(2+1)D########################################################
class R2P1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, first_conv=False):
        super(R2P1D, self).__init__()
        
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        if first_conv:
            
            spatial_kernel_size = kernel_size
            spatial_stride = (1, stride[1], stride[2])
            spatial_padding = padding
            
            temporal_kernel_size = (3, 1, 1)
            temporal_stride = (stride[0], 1, 1)
            temporal_padding = (1, 0, 0)
            
            
            intermed_channels = 45
            
            
            self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                          stride=spatial_stride, padding=spatial_padding, bias=bias)
            self.bn1 = nn.BatchNorm3d(intermed_channels)
                                          
            self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                                                         stride=temporal_stride, padding=temporal_padding, bias=bias)
            self.bn2 = nn.BatchNorm3d(out_channels)
            self.relu = nn.ReLU()
        else:
            
            spatial_kernel_size =  (1, kernel_size[1], kernel_size[2])
            spatial_stride =  (1, stride[1], stride[2])
            spatial_padding =  (0, padding[1], padding[2])
            
            temporal_kernel_size = (kernel_size[0], 1, 1)
            temporal_stride = (stride[0], 1, 1)
            temporal_padding = (padding[0], 0, 0)
            
            
            intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                                               (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels))) 
                                               
            self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                                                             stride=spatial_stride, padding=spatial_padding, bias=bias)
            self.bn1 = nn.BatchNorm3d(intermed_channels)
                                               
                                               
            self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                                                              stride=temporal_stride, padding=temporal_padding, bias=bias)
            self.bn2 = nn.BatchNorm3d(out_channels)
            self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.bn1(self.spatial_conv(x)))
        x = self.relu(self.bn2(self.temporal_conv(x)))
        return x

################################################LSTM model#####################################################
class CONV_LSTM_CELL(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(CONV_LSTM_CELL, self).__init__()
        
        assert hidden_channels % 2 == 0
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4
        
        self.padding = int((kernel_size - 1) / 2)
        
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding,  bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        
        self.Wci = None
        self.Wcf = None
        self.Wco = None
    
    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc
    
    def init_hidden(self, batch_size, hidden, shape):
        self.Wci = torch.tensor(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        self.Wcf = torch.tensor(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        self.Wco = torch.tensor(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        return (torch.tensor(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                torch.tensor(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


class CONV_LSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1,bias=True):
        super(CONV_LSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.bias = bias
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = CONV_LSTM_CELL(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.bias)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i], shape=(height, width))
                    internal_state.append((h, c))
                    
                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            
            
        return x

class CONV_LSTM_UNIT(nn.Module):
    
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1,bias=True):
        super(CONV_LSTM_UNIT, self).__init__()
        self.input_channels=input_channels
        self.hidden_channels=hidden_channels
        self.kernel_size=kernel_size
        self.step=step
        self.bias=bias
        self.convlstm = CONV_LSTM(input_channels=self.input_channels, hidden_channels=self.hidden_channels, kernel_size=self.kernel_size, step=self.step)
    
    def forward(self, input):
        batch_size=input.size()[0]
        for batch in range(batch_size):
            output=self.convlstm(input[batch])
            output=output.unsqueeze(0)
            if batch==0:
                outputs=output
            else:
                outputs=torch.cat((outputs,output),0)
        return outputs

        
        


############################################3D_CNN_model########################################################
class BN_AC_CONV3D(nn.Module):

    def __init__(self, num_in, num_filter,
                 kernel=(1,1,1), pad=(0,0,0), stride=(1,1,1),  bias=False,first_conv=False,r21=False):
        super(BN_AC_CONV3D, self).__init__()
        self.first_conv=first_conv
        self.bn = nn.BatchNorm3d(num_in)
        self.relu = nn.ReLU(inplace=True)
        if r21:
            self.conv = R2P1D(num_in, num_filter, kernel_size=kernel, stride=stride, padding=pad, bias=bias, first_conv=self.first_conv)
        else:
            self.conv = nn.Conv3d(num_in, num_filter, kernel_size=kernel, padding=pad,stride=stride, bias=bias)


    def forward(self, x):
        h = self.relu(self.bn(x))
        h = self.conv(h)
        return h
##################################################CBAM model####################################################
class CHANNEL_ATTENTION(nn.Module):

    def __init__(self, num_in, ratio=16):
        super(CHANNEL_ATTENTION, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc1   = nn.Conv3d(num_in, num_in // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv3d(num_in // 16, num_in, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SPATIAL_ATTENTION(nn.Module):
    def __init__(self, kernel_size=7):
        super(SPATIAL_ATTENTION, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
####################################################MF_unit#####################################################
class MF_UNIT_CBAM(nn.Module):

    def __init__(self, num_in, num_mid, num_out, stride=(1,1,1), first_block=False, use_3d=True):
        super(MF_UNIT_CBAM, self).__init__()
        num_ix = int(num_mid/4)
        kt,pt = (3,1) if use_3d else (1,0)
        # prepare input
        self.conv_i1 =     BN_AC_CONV3D(num_in=num_in,  num_filter=num_ix,  kernel=(1,1,1), pad=(0,0,0),first_conv=True)
        self.conv_i2 =     BN_AC_CONV3D(num_in=num_ix,  num_filter=num_in,  kernel=(1,1,1), pad=(0,0,0),first_conv=True)
        # main part
        self.conv_m1 =     BN_AC_CONV3D(num_in=num_in,  num_filter=num_mid, kernel=(kt,3,3), pad=(pt,1,1), stride=stride)
        if first_block:
            self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1,1,1), pad=(0,0,0))
        else:
            self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1,3,3), pad=(0,1,1))
        # adapter
        if first_block:
            self.conv_w1 = BN_AC_CONV3D(num_in=num_in,  num_filter=num_out, kernel=(1,1,1), pad=(0,0,0), stride=stride)
        self.ca = CHANNEL_ATTENTION(num_out)
        self.sa = SPATIAL_ATTENTION()

    def forward(self, x):

        h = self.conv_i1(x)
        x_in = x + self.conv_i2(h)

        h = self.conv_m1(x_in)
        h = self.conv_m2(h)
        
        h = self.ca(h) * h
        h = self.sa(h) * h
        if hasattr(self, 'conv_w1'):
            x = self.conv_w1(x)

        return h + x


class MF_UNIT(nn.Module):
    
    def __init__(self, num_in, num_mid, num_out, stride=(1,1,1), first_block=False, use_3d=True):
        super(MF_UNIT, self).__init__()
        num_ix = int(num_mid/4)
        kt,pt = (3,1) if use_3d else (1,0)
        # prepare input
        self.conv_i1 =     BN_AC_CONV3D(num_in=num_in,  num_filter=num_ix,  kernel=(1,1,1), pad=(0,0,0),first_conv=True)
        self.conv_i2 =     BN_AC_CONV3D(num_in=num_ix,  num_filter=num_in,  kernel=(1,1,1), pad=(0,0,0),first_conv=True)
        # main part
        self.conv_m1 =     BN_AC_CONV3D(num_in=num_in,  num_filter=num_mid, kernel=(kt,3,3), pad=(pt,1,1), stride=stride)
        if first_block:
            self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1,1,1), pad=(0,0,0))
        else:
            self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1,3,3), pad=(0,1,1))
        # adapter
        if first_block:
            self.conv_w1 = BN_AC_CONV3D(num_in=num_in,  num_filter=num_out, kernel=(1,1,1), pad=(0,0,0), stride=stride)


    def forward(self, x):
        
        h = self.conv_i1(x)
        x_in = x + self.conv_i2(h)
        
        h = self.conv_m1(x_in)
        h = self.conv_m2(h)
            
        if hasattr(self, 'conv_w1'):
            x = self.conv_w1(x)
            
        return h + x


##################################################RES2_UNIT####################################################
class RES2_UNIT(nn.Module):
    
    def __init__(self, num_in, num_mid, num_out, stride=(1,1,1), first_block=False):
        super(RES2_UNIT, self).__init__()
        num_ix=num_mid//4
        self.conv_i =  nn.Conv3d(num_in,  num_mid,  kernel_size=(1,1,1), padding=(0,0,0),stride=(1,1,1))
       
        self.conv_m1 = nn.Conv3d(num_ix*2,  num_ix, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1))
        
        self.conv_m2 = nn.Conv3d(num_ix*2, num_ix, kernel_size=(3,3,3), padding=(1,1,1),stride=(1,1,1))
        
        self.conv_m3 = nn.Conv3d(num_ix*2, num_ix, kernel_size=(3,3,3), padding=(1,1,1),stride=(1,1,1))
        
        self.conv_o =  nn.Conv3d(num_mid, num_out, kernel_size=(1,1,1), padding=(0,0,0),stride=(1,1,1))


    def forward(self, x):
        
        x=self.conv_i(x)
        num=x.size()[1]//4
        x1=x[:,0:num,:,:,:]
        x2=x[:,num:2*num,:,:,:]
        x3=x[:,2*num:3*num,:,:,:]
        x4=x[:,3*num:4*num,:,:,:]
        h1=x1
        h2=self.conv_m1(torch.cat([h1,x2],1))
        h3=self.conv_m2(torch.cat([h2,x3],1))
        h4=self.conv_m3(torch.cat([h3,x4],1))
        h=torch.cat([h1,h2,h3,h4],1)
        h=self.conv_o(h)
        return h


#################################################Net###########################################################
class MFNET_3D(nn.Module):

    def __init__(self, num_classes, batch_size=1, **kwargs):
        super(MFNET_3D, self).__init__()
        self.batch_size=batch_size
        k_sec  = {  2: 3, \
                    3: 4, \
                    4: 6, \
                    5: 3  }

        # conv1 - x224 (x16)
        conv1_num_out = 16
        self.conv1 = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv3d( 3, conv1_num_out, kernel_size=(3,5,5), padding=(1,2,2), stride=(1,2,2), bias=False)),
                    ('bn', nn.BatchNorm3d(conv1_num_out)),
                    ('relu', nn.ReLU(inplace=True))
                    ]))
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        # conv2 - x56 (x8)
        num_mid = 96
        conv2_num_out = 96
        self.conv2 = nn.Sequential(OrderedDict([
                    ("B%02d"%i, MF_UNIT_CBAM(num_in=conv1_num_out if i==1 else conv2_num_out,
                                        num_mid=num_mid,
                                        num_out=conv2_num_out,
                                        stride=(2,1,1) if i==1 else (1,1,1),
                                        first_block=(i==1))) for i in range(1,k_sec[2]+1)
                    ]))

        # conv3 - x28 (x8)
        num_mid *= 2
        conv3_num_out = 2 * conv2_num_out
        self.conv3 = nn.Sequential(OrderedDict([
                    ("B%02d"%i, MF_UNIT_CBAM(num_in=conv2_num_out if i==1 else conv3_num_out,
                                        num_mid=num_mid,
                                        num_out=conv3_num_out,
                                        stride=(1,2,2) if i==1 else (1,1,1),
                                        first_block=(i==1))) for i in range(1,k_sec[3]+1)
                    ]))

        # conv4 - x14 (x8)
        num_mid *= 2
        conv4_num_out = 2 * conv3_num_out
        self.conv4 = nn.Sequential(OrderedDict([
                    ("B%02d"%i, MF_UNIT_CBAM(num_in=conv3_num_out if i==1 else conv4_num_out,
                                        num_mid=num_mid,
                                        num_out=conv4_num_out,
                                        stride=(1,2,2) if i==1 else (1,1,1),
                                        first_block=(i==1))) for i in range(1,k_sec[4]+1)
                    ]))

        # conv5 - x7 (x8)
        self.convRes2=RES2_UNIT(conv4_num_out,conv4_num_out*2,conv4_num_out)
        num_mid *= 2
        conv5_num_out = 2 * conv4_num_out
        self.conv5 = nn.Sequential(OrderedDict([
                    ("B%02d"%i, MF_UNIT(num_in=conv4_num_out if i==1 else conv5_num_out,
                                        num_mid=num_mid,
                                        num_out=conv5_num_out,
                                        stride=(1,2,2) if i==1 else (1,1,1),
                                        first_block=(i==1))) for i in range(1,k_sec[5]+1)
                    ]))
        
        
        self.conv_lstm=CONV_LSTM_UNIT(input_channels=8, hidden_channels=[16,8], kernel_size=3, step=10)
        # final
        self.tail = nn.Sequential(OrderedDict([('bn', nn.BatchNorm3d(conv5_num_out)),('relu',nn.ReLU(inplace=True))]))

        self.globalpool = nn.Sequential(OrderedDict([
                        ('avg', nn.AvgPool3d(kernel_size=(8,7,7),  stride=(1,1,1))),
                        # ('dropout', nn.Dropout(p=0.5)), only for fine-tuning
                        ]))
        self.classifier = nn.Linear(conv5_num_out, num_classes)


        #############
        # Initialization
        initialize.xavier(net=self)
        

    def forward(self, x):
        #assert x.shape[2] == 16
        
        h = self.conv1(x)   # x224 -> x112
        h = self.maxpool(h) # x112 ->  x56

        h = self.conv2(h)   #  x56 ->  x56
        h = self.conv3(h)   #  x56 ->  x28
        h = self.conv4(h)   #  x28 ->  x14
        h = self.convRes2(h)
        h = self.conv5(h)   #  x14 ->   x7
        

        h=self.conv_lstm(h)
        
        h = self.tail(h)
        h = self.globalpool(h)

        h = h.view(h.shape[0], -1)
        h=h.squeeze()
        h=h.view(768)

        return h

if __name__ == "__main__":
    import torch
    logging.getLogger().setLevel(logging.DEBUG)
    # ---------
    net = MFNET_3D(num_classes=100, pretrained=False,batch_size=1)
    data = torch.autograd.Variable(torch.randn(1,3,16,224,224))
    output= net(data)
    print (output.shape)


