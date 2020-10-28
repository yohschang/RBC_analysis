# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 20:48:06 2020

@author: YX
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage as sk
import skimage.morphology as sm

def pad_output(crop_obj_rimap, y_mass , x_mass , outputsize):
    opsize = outputsize//2
    crop_obj_rimap = np.pad(crop_obj_rimap,((opsize,opsize),(opsize,opsize)),"constant")
    x_mass = x_mass+opsize
    y_mass = y_mass+opsize
    crop_obj_rimap = crop_obj_rimap[x_mass-opsize:x_mass+opsize,y_mass-opsize:y_mass+opsize]  # crop object with image center being coordinate of center of mass

    return crop_obj_rimap

filepath = r"D:\data\2020-10-17\rbc2\phi\rbc1"
db = pd.read_pickle(filepath+"\\rbc1.pickle")
phi_stack = []

for i in range(db["rbc_array"].size):
    phimap = db["rbc_array"].iloc[i]
    # phi_stack.append(phimap)
    phase_img = ((phimap.real+0.5)**8).astype(np.uint8)
    [row,column] = np.meshgrid(np.arange(phase_img.shape[1]), np.arange(phase_img.shape[0]))

    val1 , phase_img = cv2.threshold(phase_img,0,255,cv2.THRESH_OTSU)   #otsu binary phase img
    phase_img = sk.filters.median(sm.closing(phase_img,sm.disk(3)))
    objs_edge,__ = cv2.findContours(phase_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    # separate different edge
    x, y, w, h = cv2.boundingRect(objs_edge[0])     # output each object region
    cv2_center = (x+w//2,y+h//2)
    
    crop_sz = max(w,h)

    phimap[phase_img == 0] = 0
    x_ct_mass = int(round(sum(phimap.real[phase_img == 255]*row[phase_img == 255])/sum(phimap.real[phase_img == 255]))) #calculate x coodinate of center of mass
    y_ct_mass = int(round(sum(phimap.real[phase_img == 255]*column[phase_img == 255])/sum(phimap.real[phase_img == 255]))) #calculate y coodinate of center of mass
    crop_x = [x_ct_mass-crop_sz//2,x_ct_mass+crop_sz//2]
    crop_y = [y_ct_mass-crop_sz//2,y_ct_mass+crop_sz//2]
    recrop_phi = pad_output(phimap, cv2_center[0] , cv2_center[1] , crop_sz)
    # recrop_phi = phimap[cv2_center[1]-crop_sz//2:cv2_center[1]+crop_sz//2,cv2_center[0]-crop_sz//2:cv2_center[0]+crop_sz//2]
    phi_stack.append(recrop_phi)
    # plt.imshow(phase_img[crop_y[0]:crop_y[1],crop_x[0]:crop_x[1]],vmin = 0,vmax = 10)
    # plt.imshow(recrop_phi.real,cmap = "jet" )
    # plt.title(str(i))
    # plt.show()
    np.save(r"D:\data\2020-10-17\rbc2\phi\rbc1\re_crop",phi_stack)
