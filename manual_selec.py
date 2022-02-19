# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:51:25 2021

@author: YX
"""
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
from glob import glob
import cv2
import os
from skimage import filters
import skimage.morphology as sm
import skimage as sk

def mouse_func(event,x,y,flags,param):
    global next_f

    if event == cv2.EVENT_RBUTTONDOWN:
        next_f = True
        
def get_edge(phimap):

    phase_img = ((phimap.real+1)**8).astype(np.uint8)
    thres = filters.threshold_otsu(phase_img)
    val1 , phase_img = cv2.threshold(phase_img,thres*0.5,255,cv2.THRESH_BINARY)   #otsu binary phase img

    phase_img = sk.filters.median(sm.closing(phase_img,sm.disk(3)))
    # phase_img = sm.erosion(phase_img,sm.disk(2))
    objs_edge, __ = cv2.findContours(phase_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    # separate different edge
    return objs_edge, phase_img
    
def add_2pi(img,mode):
    phimap = img.copy()
    phimap.real[phimap.real < -0.5] += 2*np.pi
    _,phase_img = get_edge(phimap.real.copy())
    # plt.imshow(phase_img)
    # plt.show()
    if mode == 1:
        img.real[phase_img > 0] += np.pi*2
    elif mode == 2:
        img.real[(phase_img > 0 )&(img.real < -0.1)] += (2*np.pi)

    return img

j = 1

while j <= 18:
# for j in [0,1,5,7,9,10]:
    path = r"C:\Users\YX\Desktop\temp\8\phi"+"\\"+str(j)+r"\rbc\re_crop.npy"
    img_stack = np.load(path , allow_pickle=True)
    i = 0
    # for count , frame in enumerate(sorted(glob(path+"\*.npy"),key=os.path.getmtime)):
    
    del_list = []
    while i < len(img_stack):
        img = img_stack[i]
    
        img_r = img.real
        
        
        boxes = []
    
        cv2.namedWindow('Frame',2)
        cv2.resizeWindow('Frame', 200,200) 
        
        stay = True
        plt.imshow(img_r[:,:] , cmap = "jet",vmin = 0)
        plt.colorbar()
        # plt.title(str(np.sum(img_r[img_r > 0])))
        plt.title(str(i) + "/" + str(len(img_stack)-1))
        plt.show()
        
        # if i == 0:
        #     ud = input("set u d :").split(",")
        #     u,d = [int(x) for x in ud]
    
        while stay:
            next_f  = False
            key = cv2.waitKey(100) & 0xff

            cv2.setMouseCallback('Frame',mouse_func)
            cv2.imshow('Frame',img_r)
            # print(key)
            if key == ord('k') or next_f  == True :
                # 第十一步：选择一个区域，按s键，并将tracker追踪器，frame和box传入到trackers中
                [x,y,w,h]= cv2.selectROI('Frame', img_r, fromCenter=False,showCrosshair=True)
                img = img[y:y+h,x:x+w]
                img_stack[i] = img
                i+=1
                break
       
                
            elif  key == ord('u'):
                i+=1
                break
            
            elif key == ord('j'):
                i-=1
                u = 0; d = -1
                break
            
            elif key == ord('d'):
                print('delete')
                del_list.append(i)
                i+=1
                # os.remove(frame)
                # file_list.remove(frame)
                
                break
            
            elif key == ord('q'):
                i = len(img_stack)
                break
    
            elif key == ord('a'):
                mode = 2
                img_stack[i] = add_2pi(img,mode)
                # i+=1
                break
            elif key == ord('z'):
                mode = 1
                img_stack[i] = add_2pi(img,mode)
                # i+=1
                break
    
    cv2.destroyAllWindows()
    
    # img_stack = np.load(path , allow_pickle=True)
    if len(del_list) > 0:
        img_stack = np.delete(img_stack , np.unique(del_list))
    
    np.save(path , img_stack)
    print(path + "-- done ")
    
    nextstep = input("proceed next ? [y/n] : ")
    if nextstep == "n":
        break
    elif nextstep == "r":
        pass
    else:
        j+=1
        pass