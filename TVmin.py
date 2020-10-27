# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:26:25 2020

@author: YX
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import img_as_float
import cupy


class TvMin(object):
    
    def __init__(self , to =0.15 , lamb = 0.015 , iter = 10 , verbose = False):
        self.to = to
        self.lamb = lamb
        self.iterationNumber = iter
        self.verbose = verbose
        self.resultImage = np.array([])
    
    def getResultImage(self):
        return self.resultImage

    def getInputImage(self):
        return self.inputImage
    
    def setInputImage(self, inImage):                       
        self.inputImage = inImage
    
    def minimize(self):
        
        p= cupy.array(self.gradient(0*self.inputImage))

        for ind in range(0, self.iterationNumber):
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
            # print(self.resultImage)
            # print(self.norm2_3d(self.resultImage - p_prev))
        
    
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
