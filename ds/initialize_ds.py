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
            name = name.replace('module.','')
         
         
            if name in net_state_keys:
            
                dst_param_shape = net.state_dict()[name].shape
                if param.shape == dst_param_shape:
                    print(name)
                    net.state_dict()[name].copy_(param.view(dst_param_shape))
                
                    net_state_keys.remove(name)
    else:
        for state,name_list in state_dict.items():
       
            if state=='state_dict':
                for name,param in name_list.items():
                     name=name.replace('module.','')
                     if name in net_state_keys:
                        
                        dst_param_shape = net.state_dict()[name].shape
                        if param.shape == dst_param_shape:
                            print(name)
                    
                            net.state_dict()[name].copy_(param.view(dst_param_shape))
                            
                            net_state_keys.remove(name)
    

        # indicating missed keys
    if net_state_keys:
        return net_state_keys



def init_from_dict_c3d(net,model_path, device_state='cpu'):
    state_dict=torch.load(model_path,map_location=device_state)

        # customized partialy load function
    net_state_keys = list(net.state_dict().keys())

    corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }

        


    for name, param in state_dict.items():

            if name not in corresp_name:
                continue
            else:
                true_name=corresp_name[name]
            
            if true_name in net_state_keys:
                
            
                dst_param_shape = net.state_dict()[true_name].shape
                if param.shape == dst_param_shape:
                    print(true_name)
                    #print(net.state_dict()[true_name])
                    net.state_dict()[true_name].copy_(param.view(dst_param_shape))
                    
                    #print(net.state_dict()[true_name])

                
                    net_state_keys.remove(true_name)



