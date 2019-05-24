# -*- coding: utf-8 -*

    
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
            frame1 = cv2.imread(image_name[0])
            prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame1)
            hsv[...,1] = 255
    
            for i in image_name:
                print(i)
                frame2 = cv2.imread(i)
                next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    #cv2.calcOpticalFlowFarneback(prev, next, pyr_scale, levels, winsize, iterations, poly_n,
    #poly_sigma, flags[)
    #pyr_scale – parameter, specifying the image scale (<1) to build pyramids for each image;
    #pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the
    #previous one.
    
    #poly_n – size of the pixel neighborhood used to find polynomial expansion in each pixel;
    #typically poly_n =5 or 7.
    
    #poly_sigma – standard deviation of the Gaussian that is used to smooth derivatives used
    #as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for
    #poly_n=7, a good value would be poly_sigma=1.5.
    
    #flag 可选 0 或 1,0 计算快，1 慢但准确
    
                flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 1)
    
    #cv2.cartToPolar Calculates the magnitude and angle of 2D vectors.
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                #rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
                save_name=save_path+'/'+str(i)
                print(save_name)
                cv2.imwrite(save_name,rgb)
                prvs = next
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean the hiding file')
    parser.add_argument('target_folder', help='Path to folder where to be clean.')
    args = parser.parse_args()
    conv_optical_flow(args.target_folder)
