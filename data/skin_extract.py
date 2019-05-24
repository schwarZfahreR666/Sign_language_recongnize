# -*- coding: utf-8 -*
""" Created on Mon Jan 27 12:28:46 2014

@author: duan """
    
import cv2
import numpy as np
import os
import argparse
def mkdir(path):
    
    folder = os.path.exists(path)
    
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
def conv_optical_flow(target_dir):
    rootPath = os.getcwd()
    target_path = os.path.abspath(target_dir)
    mkdir(target_path.replace("cdata", "pdata"))
    os.chdir(target_path)
    image_file = os.listdir(os.getcwd())
    for i in image_file:
        image_path = os.path.join(target_path,i)
        mkdir(image_path.replace("cdata", "pdata"))
        os.chdir(image_path)
        image_file2 = os.listdir(os.getcwd())
        for n in image_file2:
            image_path2 = os.path.join(image_path,n)
            mkdir(image_path2.replace("cdata", "pdata"))
            os.chdir(image_path2)
            print(os.getcwd())
            data_dir=image_path2
            save_path=data_dir.replace("cdata", "pdata")

            image_name = os.listdir(os.getcwd())
            image_name.sort()
            
    
            for i in image_name:
                print(i)
                frame2 = cv2.imread(i)
                
                rgb=hand_de(frame2)
                save_name=save_path+'/'+str(i)
                print(save_name)
                cv2.imwrite(save_name,rgb)
                prvs = next

def hand_de(img):
    px = img[150,200]
    blue = img[150,200,0]
    green = img[150,200,1]
    red = img[150,200,2]
    img[150,200] = [0,0,0]
    blue = img.item(100,200,0)
    green = img.item(100,200,1)
    red = img.item(100,200,2)
    img.itemset((100,200,1),255)
    green = img.item(100,200,1)
    rows,cols,channels = img.shape
    imgSkin = np.zeros(img.shape, np.uint8)
    imgSkin = img.copy()
    
    for r in range(rows):
        for c in range(cols):
            B = img.item(r,c,0)
            G = img.item(r,c,1)
            R = img.item(r,c,2)
            skin = 0
            
            if (abs(R - G) > 15) and (R > G) and (R > B):
                if (R > 95) and (G > 40) and (B > 20) and (max(R,G,B) - min(R,G,B) > 15):
                    skin = 1
                elif (R > 220) and (G > 210) and (B > 170):
                    skin = 1
            
            if 0 == skin:
                imgSkin.itemset((r,c,0),0)
                imgSkin.itemset((r,c,1),0)
                imgSkin.itemset((r,c,2),0)
    return imgSkin
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean the hiding file')
    parser.add_argument('target_folder', help='Path to folder where to be clean.')
    args = parser.parse_args()
    conv_optical_flow(args.target_folder)
