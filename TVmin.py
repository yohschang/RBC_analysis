# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:26:25 2020

reference : https://github.com/gokhangg
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_float
import cupy
from tqdm import tqdm
from skimage import filters


class TvMin(object):
    
    def __init__(self ,dF_3D_2 = cupy.array([]), to =0.15 , lamb = 0.1 , iteration = 100 , verbose = False):
        self.to = to
        self.lamb = lamb
        self.iterationNumber = iteration
        self.verbose = verbose
        self.resultImage = np.array([])
        self.dF_3D_2 = dF_3D_2
        self.error = 0.002
        self.dn_3D_bi = cupy.array([])
    
    def getResultImage(self):
        return self.resultImage

    def getInputImage(self):
        return self.inputImage
    
    def setInputImage(self, inImage):                       
        self.inputImage = inImage
    
    def minimize(self):
        
        # self.create_3d_mask()
        
        p= cupy.array(self.gradient(0*self.inputImage))
        self.iterationNumber += 10
        for ind in tqdm(range(0, self.iterationNumber),desc = "tvmin"):
            if self.verbose:
                print("Itertion: ",ind)
            if ind < 10:
                lamb = 0.5
            else:
                lamb = self.lamb
                
            midP = self.divergence(p) - self.inputImage/lamb
            psi=cupy.array(self.gradient(midP))
            r = self.getSquareSum(psi)
            p = (p + self.to*psi)/(1 + self.to*r)
            self.resultImage = (self.inputImage - self.divergence(p)*lamb)
            self.resultImage = self.positive_constrain(self.resultImage,self.dF_3D_2)
            if ind == 9 :
                n_3D = cupy.asnumpy(self.resultImage).astype(float)
                otsu_val = filters.threshold_otsu(n_3D)
                n_3D_bi = np.zeros_like(n_3D)
                n_3D_bi[n_3D >= otsu_val]= 1
                self.dn_3D_bi = cupy.asarray(n_3D_bi) 
            elif ind > 9:
                self.resultImage[cupy.equal(self.dn_3D_bi , 0)] = 0
            
            ##critirian
            # if self.norm2_3d(self.resultImage - p_prev) < self.error :
            #     print(ind)
            #     break
    
    def create_3d_mask(self):
                
        p= cupy.array(self.gradient(0*self.inputImage))
        lamb = 0.5
        for ind in range(0, 10):
            if self.verbose:
                print("Itertion: ",ind)
            if ind == 0:
                p_prev = cupy.zeros_like(self.inputImage)
            # print(p_prev.shape)
            else:
                p_prev = self.resultImage
            midP = self.divergence(p) - self.inputImage/self.lamb
            psi=cupy.array(self.gradient(midP))
            r = self.getSquareSum(psi)
            p = (p + self.to*psi)/(1 + self.to*r)
            self.resultImage = (self.inputImage - self.divergence(p)*self.lamb)
            self.resultImage = self.positive_constrain(self.resultImage,self.dF_3D_2)



                
    
    def positive_constrain(self,dn_3D,dF_3D_2):
        n_med = 1.334
        wavelength = 532
        k2 = (2*np.pi*n_med/wavelength)**2
        n_med2 = n_med**2
        # dF_3D = cupy.multiply(cupy.subtract(cupy.divide(cupy.multiply(dn_3D,dn_3D),n_med2),1),-k2)
        # dF_3D = cupy.fft.fftn(dF_3D)
        # dF_3D[cupy.not_equal(dF_3D_2,0)] = dF_3D_2[cupy.not_equal(dF_3D_2,0)]
        # dF_3D   = cupy.fft.ifftn(dF_3D)    
        # dn_3D   = cupy.multiply(cupy.sqrt(cupy.add(cupy.divide(dF_3D,-k2), 1)), n_med)
        
        # dn_3D =  cupy.fft.fftshift(dn_3D);
        dn_3D[cupy.less(cupy.real(dn_3D),n_med)] = n_med+1j*cupy.imag(dn_3D[cupy.less(cupy.real(dn_3D),n_med)])
        dn_3D[cupy.less(cupy.imag(dn_3D),0)]     = cupy.real(dn_3D[cupy.less(cupy.imag(dn_3D), 0)])


        return dn_3D
    
    def norm2_3d(self,x):
        return(cupy.sqrt(cupy.sum(x**2)))
    
    def gradient(self,inImage):
        imageDimension = len(inImage.shape)
        result = []
        for ind in range(imageDimension - 1, -1,-1):
            result += [self.forwardDerivative(inImage.swapaxes(imageDimension - 1, ind)).swapaxes(imageDimension - 1, ind)]
        return result

    def divergence(self, inImage):
        imageDimension = len(inImage.shape) - 1
        summation = 0
        for ind in range(imageDimension - 1, -1, -1):
            summation += self.backDerivative(inImage[imageDimension - 1 - ind].swapaxes(imageDimension - 1, ind)).swapaxes(imageDimension - 1, ind)
        return summation

    @staticmethod  
    def getSquareSum(inImage):
        imageDimension = len(inImage.shape) - 1
        summation = 0
        for ind in range(0,imageDimension):
            summation += inImage[ind]**2
        return cupy.sqrt(summation)
    
    @staticmethod    
    def forwardDerivative(In):
         d = 0*In
         d[:-1] = In[1:] - In[:-1]
         return d
    
    @staticmethod  
    def backDerivative(In):
        d = 0*In
        d[1:] = In[1:] - In[:-1]
        return d  
    
# aa = np.ones((100,100,100))+100
# noise = np.random.normal(0, 20, 100*100*100).reshape(100,100,100)
# aa = noise+aa
if __name__ == "__main__":
    load_file_name = 'noi_image'
    noi_image = sio.loadmat(load_file_name)
    noi_img = noi_image['noi_img'][50:100,25:75]
    # noi_img = cupy.ones((50,50))+100
    # noise = cupy.random.normal(1,10,50*50).reshape(50,50)
    # img_3d = noi_img + noise
    # img_3d = noi_img[...,cupy.newaxis]
    # img_3d = np.tile(img_3d , (1,1,50))
    aa = cupy.array(noi_img) #+ 1j*img_3d
    # print(aa)
    # print("___________________________")
    tv = TvMin()
    tv.setInputImage(aa)
    tv.minimize()
    re = tv.getResultImage()
    # print(re)
    print("___________________________")
    print("origin : " + str(np.mean(aa))+" , "+str(np.std(aa)))
    print("after : " + str(np.mean(re))+" , "+str(np.std(re)))
    
    aa = cupy.asnumpy(aa)
    re = cupy.asnumpy(re)
    plt.imshow(aa)
    plt.colorbar()
    plt.show()
    plt.imshow(re)
    plt.colorbar()
    plt.show()
