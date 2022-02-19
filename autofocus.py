# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:01:49 2020

@author: YX

## reason why freq domian coordinate :　np.fft.fftfreq
## fftfreq def:https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.fft.fftfreq.html
which can be discribe as "linspace(-1/2/pitch,1/2/pitch-1/l,n)" (period range from -1/2 to 1/2)
where pitch is sampling freq (equal to "d" in fftfreq)
and n is total datapoint
and l is n*pitch(reallenth)
in the situation below, since sampling freq == 1 pixel therefore d==1
so the coordinate in spatial freq domain are fftfreq(size)

ref : https://github.com/thu12zwh/band-extended-angular-spectrum-method/blob/master/band-extended%20angular%20spectrum%20method/band_extended_ASM_submit.m
"""
import numpy as np
import time
import matplotlib.pyplot as plt
import cupy
import cv2
import pandas as pd
import time
from re_crop import recrop
from tqdm import tqdm
from glob import glob

class refocus():
    def __init__(self,path,prop_range = [-100,100],edge_blank = 0,plot = True , save = True):
        self.wavelen = 0.532
        self.n_med = 1.334
        self.path = path
        self.plot = plot
        self.save = save
        self.prop_range = prop_range
        self.edge_blank = edge_blank
        
    def fft_propagate_3d(self,phimap ,d):
    
        km = (2 * np.pi * self.n_med) / self.wavelen
        kx = (cupy.fft.fftfreq(phimap.shape[0]) * 2 * np.pi).reshape(-1, 1)
        ky = (cupy.fft.fftfreq(phimap.shape[1]) * 2 * np.pi).reshape(1, -1)
        root_km = km**2 - kx**2 - ky**2
        rt0 = (root_km > 0)
        fstemp = cupy.exp(1j * (np.sqrt(root_km * rt0) - km) * d) * rt0
        return cupy.fft.ifft2(phimap* fstemp)

        
    def tamura_coefficient(self , img):
        return cupy.std(img,ddof = 1)**2/cupy.abs(cupy.mean(img))
    
    def run(self):
        # phi_stack= np.load(self.path+r"\re_crop.npy",allow_pickle=True)
        phi_stack= np.load(self.path+r"\re_crop.npy",allow_pickle=True)

        for count, i in tqdm(enumerate(phi_stack)):
            if i.shape[0]>70:
                i = i[10:-10,10:-10]
                
                
            tamu= []
            phi = cupy.array(i)
            length = cupy.arange(self.prop_range[0],self.prop_range[1],1)
            tamu = []
            for d in length:
                prop_phi = self.fft_propagate_3d(np.fft.fft2(phi.copy()), d)
                tamu.append(float(self.tamura_coefficient(prop_phi.real)))
            tamu = np.array(tamu)
            if count == 0:
                plt.plot(cupy.asnumpy(length) ,tamu)
                plt.ylabel("Tamura coefficient")
                plt.xlabel("propagation distance (z)")
                plt.show()
                # proceed = input("proceed?")
                # if proceed == "":
                #     pass
                # else:
                #     self.prop_range = [int(proceed.split(",")[0]),int(proceed.split(",")[1])]
            max_tamu = int(np.where(tamu == max(tamu))[0])
            prop_phi = self.fft_propagate_3d(np.fft.fft2(phi) ,length[max_tamu])
            prop_phi = cupy.asnumpy(prop_phi)
            # phi_stack[count] = prop_phi[self.edge_blank:-1-self.edge_blank,self.edge_blank:-1-self.edge_blank]
            phi_stack[count] = prop_phi
            
            if self.plot:
                # ax[0].imshow(i.real,cmap = "jet" , vmin = 0 , vmax = 3)
                # ax[0].set_title("z = 0")
                plt.imshow(prop_phi.real ,cmap = "jet",  vmin = 0)
                plt.title("z = "+str(length[max_tamu]))
                plt.colorbar()
                plt.show()
        re_crop = recrop(phi_stack , None , "none" ,  mask = True ,save = False ,  plot =False, region_adj=3 )
        phi_stack , elp_para = re_crop.re_crop()
  
        if self.save:
            np.save(self.path + r"\re_crop(autofocus).npy",phi_stack)
            np.save(self.path+"//elp_para" , elp_para )

if __name__ == "__main__":
    # import time
    # t1 = time.time()
    path = r"C:\Users\YX\Desktop\temp\8\phi"
    filefolder = glob(path+"/*")
    for p in filefolder:
        refocus_c = refocus(p+ "/rbc",prop_range = [-15,15],edge_blank = 1,save =True,plot =False)
        refocus_c.run()

    # t2 = time.time()
    # print(t2-t1)

#%% for simulation 
# for j in [1]:
#     for i in range(0,3): 
#         pp = str(1010)
#         try:
#             # path = r"D:\data\2021-05-12"+"\\"+str(j)+"\\phi"+"\\"+str(i)+"\\rbc"
#             if i == 0:
#                 path = r"E:\lab\CODE\recon_verify\proj\contrast_vari(phase&rotate)\20210429_"+pp
#             elif i == 1:
#                 path = r"E:\lab\CODE\recon_verify\proj\phase_vari(+rotate)\20210429_"+pp
#             elif i == 2:
#                 path = r"E:\lab\CODE\recon_verify\proj\warp_rand\20210429_"+pp
#             refocus_c = refocus(path,prop_range = [0,1],edge_blank = 1,save =True,plot =False)
#             refocus_c.run()
#         except FileNotFoundError:
#             pass
    