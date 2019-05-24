

import os
from PIL import Image
from torchvision import transforms
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import random



IMG_SIZE=224    
process=transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor()
    
])


class Image_Dataset(Dataset):
    def __init__(self, path_txt , clip_len=16,action='train'):
       
        self.clip_len=clip_len
        root=open(path_txt)
        path_list=root.readlines()
        file_s=[]
        label_s=[]
        self.file_list=[]
        self.label_list=[]
        self.process=process
        for file in path_list:
            path=file.split(' ')[0]
            label=file.split(' ')[1]
            file_s.append(path)
            label_s.append(label)
            
        label_s=np.array(label_s).astype(np.int64)
        video_nums=len(file_s)
        
        self.file_list=file_s
        self.label_list=label_s
    def __getitem__(self, index):
      
        imgs=sorted(os.listdir(self.file_list[index]))
        start_offset=len(imgs)-self.clip_len
        if start_offset>=0:
            start_index=random.randint(0,start_offset)
            for img_index in range(start_index,start_index+self.clip_len):
            
                img=Image.open(os.path.join(self.file_list[index],imgs[img_index]))
            
                data=self.process(img)
                data=data.unsqueeze(1)
                if img_index==start_index:
                    data_s=data
                else:
                    data_s=torch.cat([data_s,data],1)
        else:
            for img_index in range(0,start_offset+self.clip_len):
            
                img=Image.open(os.path.join(self.file_list[index],imgs[img_index]))
            
                data=self.process(img)
                data=data.unsqueeze(1)
                if img_index==0:
                    data_s=data
                else:
                    data_s=torch.cat([data_s,data],1)
            for re_index in range(abs(start_offset)):
                data_s=torch.cat([data_s,data],1)
                       
        label=self.label_list[index]
        return data_s,label
            
    def __len__(self):

        return len(self.file_list)
    def deal_with():
        pass




