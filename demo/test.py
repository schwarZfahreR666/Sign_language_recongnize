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
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from mfnet_3d import MFNET_3D
from imgloader import Image_Dataset
import cv2

result = {'0':'glasses','65':'搅拌'}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

Model_path='models/trained_model1.pkl'





model = MFNET_3D(num_classes=100,batch_size=1)

criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
if torch.cuda.device_count() > 1:
  model = nn.DataParallel(model)
model.to(device)
criterion.to(device)

initialize.init_from_dict(model,Model_path,device,Flags=False)
print('loaded pretrained model')

model.eval()
test_dataloader  = DataLoader(Image_Dataset('test.list'), batch_size=1, num_workers=4)

test_size = len(test_dataloader.dataset)




for inputs, labels,imgs in tqdm(test_dataloader):
        # move inputs and labels to the device the training is taking place on
        inputs = Variable(inputs, requires_grad=True).to(device)
        labels = Variable(labels).to(device)
            

                
        with torch.no_grad():
            outputs = model(inputs)
                        
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        loss = criterion(outputs, labels)
        mm = int(preds)
        
        text = result[str(mm)]

        
        for img in imgs:
          
          frame = cv2.imread(img[0])
          cv2img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
          pilimg = Image.fromarray(cv2img)
          draw = ImageDraw.Draw(pilimg)
          font = ImageFont.truetype("/Users/zr/Desktop/demo/simhei.ttf",20,encoding = "utf-8")
          draw.text((0,0),text,(255,0,0),font=font)
          frame = cv2.cvtColor(np.array(pilimg),cv2.COLOR_RGB2BGR)
          #cv2.putText(frame , text , (40,50) , cv2.FONT_HERSHEY_PLAIN , 2.0, (0,0,255),2)
          cv2.imshow('result',frame)
        

          if cv2.waitKey(1) & 0xFF == ord('q'):
              break

    
        
            

        
    
    

