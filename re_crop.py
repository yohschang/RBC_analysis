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
from skimage import filters
import skimage.morphology as sm
from skimage.morphology import disk
from scipy.ndimage import rotate
from glob import glob
import os

max_el = []
class recrop:
    def __init__(self,filepath ,filename , readmod ,save = False , plot = True , mask = False , region_adj = 5):
        self.filepath = filepath 
        self.region_adj = region_adj
        self.save  = save
        self.plot = plot
        self.mask = mask
        self.filename = filename
        self.read_mod = readmod
        self.prev_sum = 0
        self.cross180 = 0
        self.avg_phi = []
        self.prev_y = 0
        self.prev_ang = 0
        
        self.elp_para = []


    def pad_output(self , crop_obj_rimap, y_mass , x_mass , outputsize):
        opsize = outputsize//2
        crop_obj_rimap = np.pad(crop_obj_rimap,((opsize,opsize),(opsize,opsize)),"constant")
        x_mass = x_mass+opsize
        y_mass = y_mass+opsize
        crop_obj_rimap = crop_obj_rimap[x_mass-opsize:x_mass+opsize,y_mass-opsize:y_mass+opsize]  # crop object with image center being coordinate of center of mass
    
        return crop_obj_rimap
    
    def read_phistack(self):
        phistack = []
    
        if self.read_mod == "np" : #read npy
            filepath = self.filepath + "\\"+self.filename+".npy"
            phistack  = np.load(filepath,allow_pickle=True)
            
        elif self.read_mod == "pd": #read pickle
            filepath = self.filepath + "\\"+self.filename+".pickle"
            db = pd.read_pickle(filepath)
            for i in range(db["rbc_array"].size):
                phistack.append(db["rbc_array"].iloc[i])
                
        elif self.read_mod == "multi_np":
            for i in sorted(glob(self.filepath+"/*.npy"), key = os.path.getmtime):
                # print(i)
                phistack.append(i)
                
        elif self.read_mod == "single_np":
            phistack = np.load(self.filepath)

            
        else:
            return self.filepath
                
        return phistack
    
    def find_ellipse_angle(self,contour):    # calculate the angle in order to rotate back to verticle
        ellipse = cv2.fitEllipse(contour)
        centers ,radius  ,angle = ellipse
        cx,cy = int(round(centers[0])),int(round(centers[1]))
        ax1,ax2 =  int(round(radius[0])),int(round(radius[1]))
        center = (cx,cy)
        axes = (ax1,ax2)
        # print(center ,radius  ,angle)
        return center,axes,round(angle) 
    
    def get_edge(self, phimap):

        
        phase_img = ((phimap.real+1)**8).astype(np.uint8)
        thres = filters.threshold_otsu(phase_img)

        ##normal
        val1 , phase_img = cv2.threshold(phase_img,thres*0.3,255,cv2.THRESH_BINARY)   #otsu binary phase img
        phase_img = sk.filters.median(sm.closing(phase_img,sm.disk(1)))

        ##for simulate testing
        # phase_img = np.zeros_like(phimap).astype(np.uint8)
        # phase_img[phimap.real > 1 ] = 1
        # plt.imshow(phase_img)
        # plt.show()
        
        objs_edge, _ = cv2.findContours(phase_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    # separate different edge
        
        return objs_edge, phase_img
    
    
    def remove_2pi(self,img):
        # img_r = filters.median(img.real)
        mask = np.zeros_like(img, dtype=np.uint8)
        mask_sum = mask.copy()
        
        blank = 100
        # while blank > 15 and count < 10:.
        kernel = np.ones((5,5), np.uint8)
        for count in range(10):
            if blank > 15 :
                mask_sum += mask
                x_gra , y_gra = np.gradient(img_r)
                gradient = np.abs(x_gra) + np.abs(y_gra)
                x_pos , y_pos = np.where((gradient > 2) & (gradient < np.pi*1.5))
                position = np.vstack((y_pos , x_pos)).T
                if len(position) == 0:
                    position = np.array([0,0])[np.newaxis,:]
                # print(position)
                mask = cv2.fillPoly(mask , pts = [position], color=(255))
                mask = cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernel)
                
                mask[img>0] = 0
                img_r[mask == 255] += 2*np.pi
                            
                blank = np.sum(mask>0)
                if count == 0 and blank < 50:
                    img[mask == 255] = 0
                    break
            else:
                break
        mask_sum[mask_sum > 0] = 255
        img[mask > 0] += 2*np.pi
    
        # mask = cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernel)
        # img[mask == 255] = 0
        return img 
    
    def scale_contour(self, contour, scale):
        moments = cv2.moments(contour)
        midX = int(round(moments["m10"] / moments["m00"]))
        midY = int(round(moments["m01"] / moments["m00"]))
        mid = np.array([midX, midY])
        
        mid = np.array([midX, midY])
        contour = contour - mid
        contour = (contour * scale).astype(np.int32)
        contour = contour + mid
        return midX , midY , contour
    
    
    def rect_mask(self , x,y,w,h , phimap):
        x0 = [0,x-10][x-10 > 0]
        y0 = [0,y-10][y-10 > 0]
        x1 = [3072 , x+w+10][x+w+10 < 3072] 
        y1 = [3072 , y+h+10][y+h+10 < 3072] 
        phimap[0:y0, :] = 0;      phimap[:, 0:x0] = 0
        phimap[y1:3072, :] = 0;   phimap[:, x1:3072] = 0
        return phimap

    def re_crop(self):
        
        load_phi = self.read_phistack()  # 1 : read npy , 2 : read pickle 
        phi_stack = []
        
        if self.read_mod == "single_np":
            iter_obj = [1]
        else:
            iter_obj = load_phi
        for i , phimap in enumerate(iter_obj):
            if self.read_mod == "single_np":
                phimap = load_phi
            elif self.read_mod == "multi_np":
                print(phimap)
                phimap = np.load(phimap)
            
            # phimap = self.remove_2pi(phimap)
            final_phimap = phimap.copy()
            # phimap.real[phimap.real < -0.1] += 2*np.pi
            # phimap.real = filters.median(phimap.real)
            # self.remove_2pi_v2(phimap.real)
            # if self.mask:
            
            [row,column] = np.meshgrid(np.arange(phimap.shape[1]), np.arange(phimap.shape[0]))
            
            objs_edge, phase_img = self.get_edge(phimap)
            

            # if len(objs_edge) != 0 and np.sum(phimap.real) > 500:
            if len(objs_edge) != 0:

                mask_sum = 0
                for o in objs_edge:
                    mask = np.zeros_like(phase_img)
                    mask_candi = cv2.fillPoly(mask, pts =[o], color=(255))
                    if np.sum(phimap[mask_candi == 255]) > mask_sum:
                        f_mask = mask_candi
                        mask_sum = np.sum(phimap[mask_candi == 255])
                        f_o = o
                
                
                mask = np.zeros_like(phase_img)
                midX , midY ,s_contour_small = self.scale_contour(f_o , 1)
                adj_mask = cv2.fillPoly(mask, pts =[s_contour_small], color=(255))
                                
                # plt.imshow(adj_mask)
                # plt.title(str(np.max(final_phimap.real[(adj_mask == 255)])))
                # plt.show()
                
                
                # if np.sum(adj_mask[(adj_mask == 255)& (final_phimap.real < -0.1)]) > 100:
                #     final_phimap.real[(adj_mask == 255) & (final_phimap.real < 0)] += 2*np.pi
                    # final_phimap.real[(adj_mask == 255)] += 2*np.pi
                # plt.imshow(final_phimap.real , cmap = "jet" , vmin = 0)
                # plt.colorbar()
                # plt.title("add")
                # plt.show()
                
                x, y, w, h = cv2.boundingRect(f_o)     # output each object region
                cv2_center = (x+w//2,y+h//2)
                
                crop_sz = max(w,h) + self.region_adj*2
                phimap = self.rect_mask(x, y, w, h ,final_phimap)
                

                # print(crop_sz)

                # phimap = self.remove_2pi(phimap)
  
                
                if self.mask :
                    # phimap.real = filters.median(phimap.real,disk(2))
                    phimap[mask == 0] = 0
    
                
                phi_med = phimap.real
                if np.nansum(phimap.real[adj_mask == 255]) > 0:
                    x_ct_mass = int(round(np.nansum(phi_med.real[adj_mask == 255]*row[adj_mask == 255])/np.nansum(phi_med.real[adj_mask == 255]))) #calculate x coodinate of center of mass
                    y_ct_mass = int(round(np.nansum(phi_med.real[adj_mask == 255]*column[adj_mask == 255])/np.nansum(phi_med.real[adj_mask == 255]))) #calculate y coodinate of center of mass
        
        
                crop_x = [x_ct_mass-crop_sz//2,x_ct_mass+crop_sz//2]
                crop_y = [y_ct_mass-crop_sz//2,y_ct_mass+crop_sz//2]
                # crop_x = [midX-crop_sz//2,midX+crop_sz//2]
                # crop_y = [midY-crop_sz//2,midY+crop_sz//2]
                
                recrop_phi = self.pad_output(phimap, cv2_center[0] , cv2_center[1] , crop_sz)

                if self.read_mod == "single_np":
                    plt.imshow(recrop_phi.real , cmap = "jet", vmin = 0 , vmax = 3)
                    plt.show()
                    return recrop_phi 
                
                # if abs(el_radius[0] - el_radius[1]) >= 3 :
                self.avg_phi.append(np.sum(recrop_phi.real))
                mean_phi = [0 , np.mean(self.avg_phi[:-1])][len(self.avg_phi) > 1]
    
                objs_edge , _ = self.get_edge(phimap)


                if len(objs_edge[0]) >= 6:
                    self.prev_sum = np.sum(recrop_phi.real)
                    el_center , el_radius , el_ang = self.find_ellipse_angle(objs_edge[0])
                    self.elp_para.append([el_radius[0] , el_radius[1], el_ang , np.sum(recrop_phi.real)])
      
                    phi_stack.append(recrop_phi)
                # plt.imshow(phase_img[crop_y[0]:crop_y[1],crop_x[0]:crop_x[1]],vmin = 0,vmax = 10)
                    if self.plot :
                        phi_sum = np.sum(recrop_phi.real)
                        plt.imshow(recrop_phi.real,cmap = "jet" ,vmin = 0 )
                        # plt.colorbar()
                        plt.title(str(i) + " ; " + str(int(phi_sum)))
                        plt.show()
        
        if self.save:
            if os.path.exists(self.filepath+"\\"+"rbc"):
                pass
            else:
                os.mkdir(self.filepath+"\\"+"rbc")
            np.save(self.filepath+"\\"+"rbc"+"\\re_crop",phi_stack)
            # np.save(self.filepath+"//elp_para" , self.elp_para )
        else:
            # pass
            return phi_stack, self.elp_para
            # return self.ang_list

if __name__ == "__main__" :
    # import time 
    # t1 = time.time()
    for i in range(15,18):
    # for i in [4]:
        filepath = r"C:\Users\YX\Desktop\temp\8\phi"+"\\"+str(i)
        # filepath = r"D:\data\2021-04-29\4\phi\1"
    
        re_crop = recrop(filepath , "rbc1" , "multi_np" ,  mask = False ,save = True,  plot =False, region_adj = 20)
        # (filepath ,filename , readmod ,save = False , plot = True , mask = False):
        re_crop.re_crop()
        # np.save(filepath,a)
        print(i)
    # t2 = time.time()
    # print(t2-t1)
#%%
# print(a)
# plt.plot(a)
# plt.plot(np.diff(a))

# for c , aa in enumerate(a): 
#     a[c] = min(aa , 180-aa)

# #%%
# print(a)
# # if cross180 % 2 == 0 :
#     el_ang = el_ang
# else:
#     el_ang = -(180 -el_ang)











