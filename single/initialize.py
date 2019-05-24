import json
import logging

import numpy as np
import torch


def xavier(net):
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and hasattr(m, 'weight'):
            torch.nn.init.xavier_uniform(m.weight.data, gain=1.)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight.data, gain=1.)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname in ['Sequential', 'AvgPool3d', 'MaxPool3d','AdaptiveMaxPool3d', \
                           'Dropout', 'ReLU', 'Softmax', 'BnActConv3d','Sigmoid','AdaptiveAvgPool3d'] \
             or 'Block' in classname:
            pass
        else:
            if classname != classname.upper():
                logging.warning("Initializer:: '{}' is uninitialized.".format(classname))
    net.apply(weights_init)



def init_from_dict(net, model_path, device_state='cpu',Flags=False):
    state_dict=torch.load(model_path,map_location=device_state)

        # customized partialy load function
    net_state_keys = list(net.state_dict().keys())
    if Flags:
        for name, param in state_dict.items():
            
            if name in net_state_keys:
            
                dst_param_shape = net.state_dict()[name].shape
                if param.shape == dst_param_shape:
                    net.state_dict()[name].copy_(param.view(dst_param_shape))
                
                    net_state_keys.remove(name)
    else:
        for state,name_list in state_dict.items():
        
            if state=='state_dict':
                for name,param in name_list.items():
                    # name=name.replace('module.','')
                     if name in net_state_keys:
                        print(name)
                        dst_param_shape = net.state_dict()[name].shape
                        if param.shape == dst_param_shape:
                            net.state_dict()[name].copy_(param.view(dst_param_shape))
                            net_state_keys.remove(name)
    

        # indicating missed keys
    if net_state_keys:
        return net_state_keys


