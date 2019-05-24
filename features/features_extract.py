# coding: utf-8




import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import initialize
from tensorboardX import SummaryWriter
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from mfnet_3d_features import MFNET_3D
from imgloader import Image_Dataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

Model_path='models/trained_model1.pkl'

save_name='train_features.txt'





model = MFNET_3D(num_classes=100,batch_size=1)

criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
if torch.cuda.device_count() > 1:
  model = nn.DataParallel(model)
model.to(device)
criterion.to(device)

initialize.init_from_dict(model,Model_path,device)
print('loaded pretrained model')

model.eval()
test_dataloader  = DataLoader(Image_Dataset('train.list'), batch_size=1, num_workers=4)

test_size = len(test_dataloader.dataset)

f=open(save_name,"a+")

for inputs, labels in tqdm(test_dataloader):
        # move inputs and labels to the device the training is taking place on
        inputs = Variable(inputs, requires_grad=True).to(device)
        labels = Variable(labels).to(device)
            

                
        with torch.no_grad():
            outputs = model(inputs)
        f.write(str(outputs.cpu().numpy().tolist()))
        f.write(str(labels.data.cpu().numpy()))
        f.write("\n")              
        
            
f.close()
        
    
    

