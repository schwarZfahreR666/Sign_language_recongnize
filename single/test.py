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
from mfnet_3d import MFNET_3D
from imgloader import Image_Dataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

Model_path='models/trained_model1.pkl'





model = MFNET_3D(num_classes=100,batch_size=1)

criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
if torch.cuda.device_count() > 1:
  model = nn.DataParallel(model)
model.to(device)
criterion.to(device)

initialize.init_from_dict(model,Model_path,device,Flags=True)
print('loaded pretrained model')

model.eval()
test_dataloader  = DataLoader(Image_Dataset('test.list'), batch_size=1, num_workers=4)

test_size = len(test_dataloader.dataset)


running_loss=0.0
running_corrects=0.0

for inputs, labels in tqdm(test_dataloader):
        # move inputs and labels to the device the training is taking place on
        inputs = Variable(inputs, requires_grad=True).to(device)
        labels = Variable(labels).to(device)
            

                
        with torch.no_grad():
            outputs = model(inputs)
                        
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        loss = criterion(outputs, labels)
    
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
            
total_loss = running_loss / test_size
total_acc = float(running_corrects.double() / test_size)
print('Loss:',total_loss)
print('Acc:',total_acc)
        
    
    

