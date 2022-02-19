# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 18:21:05 2021

@author: YX
"""
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
from glob import glob
import cv2
import cupy 
import skimage.morphology as sm
from scipy.optimize import curve_fit , leastsq
from skimage import filters
import skimage as sk
from scipy.ndimage import rotate

def remove_outlier(array):
    array = np.array(array)
    if len(array) <= 2:
        return np.mean(array)
    else :
        mean = np.mean(array)
        std = np.std(array)
        array = [x for x in array if x < mean+1.5*std and x > mean-1.5*std]
        return np.mean(array)

class shape_curfit:
    def __init__(self):
        self.plot = False

    def func(self,c,x):
        #a:c0 , b: c2 , c: c4 , d: r 
        c2,c4 = c
        r = self.r
        c0 = self.c0
        return np.sqrt(1-(x/r)**2)*(c0+c2*(x/r)**2+c4*(x/r)**4)
    
    def loss(self,c,x,data_y):
        return np.sqrt((self.func(c,x)-data_y)**2)
        # return np.sqrt((max(self.func(c,x))-max(data_y))**2)
    
    def fit(self,img):
        size = img.shape
        rbc_curve = img[size[0]//2, :]
        self.r = len(rbc_curve)//2
        rbc_curve_y = rbc_curve[self.r:] /2
        self.c0 =  rbc_curve_y[0] #中心厚度
        rbc_curve_x = np.arange(len(rbc_curve_y))
        # print(len(rbc_curve_x),len(rbc_curve_y))
        para = leastsq(self.loss ,[4.,0.] , args=(rbc_curve_x,rbc_curve_y))
        c2,c4 = para[0]
        
        if self.plot:
            plt.plot(rbc_curve_y)
            plt.plot(self.func(para[0] , rbc_curve_x))
            plt.show()

        return [self.c0 , c2 , c4]
    

class stack_process():
    def __init__(self , img_stack):
        self.image_stack = img_stack
        self.max_radius = []
        self.radius_diff = []
        self.curve_fit = shape_curfit()
        self.distribute_var = []
        self.curvature = []
        
    
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
        val1 , phase_img = cv2.threshold(phase_img,thres*0.8,255,cv2.THRESH_BINARY)   #otsu binary phase img
        phase_img = sk.filters.median(sm.closing(phase_img,sm.disk(1)))
        objs_edge, __ = cv2.findContours(phase_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    # separate different edge
    
        return objs_edge

    def classifier(self):
        objs_edges = []
        for i in self.image_stack:
            objs_edge = self.get_edge(i)
            objs_edges.append(objs_edge)
            el_center , el_radius , el_ang = self.find_ellipse_angle(objs_edge[0])
            self.max_radius.append(max(el_radius))
            self.radius_diff.append(abs(np.diff(el_radius)))
            
        sort_rd_diff = np.unique(np.sort(self.radius_diff))
        min_axis_diff = sort_rd_diff[0]

        mean_diameter  =  np.mean(self.max_radius)
        flat_frame = np.where(self.radius_diff <= sort_rd_diff[1])[0]
        uprit_frame = np.where(self.radius_diff >= sort_rd_diff[-2])[0]
        
        # calculate max thickness
        thickness = [self.max_radius[x] - self.radius_diff[x] for x in  uprit_frame]
        max_thickness = remove_outlier(thickness)
        # np.mean(thickness) , np.std(thickness) , len(thickness)
        
        if min_axis_diff >= 3:
            return mean_diameter, min_axis_diff, max_thickness
        
        # print(flat_frame)
        # print(uprit_frame)
        for i in flat_frame:
            img = self.image_stack[i]
            size = img.shape
            x, y, w, h = cv2.boundingRect(objs_edges[i][0])
            # img = np.roll(img , size[1]//2 - (x+w//2), axis = 1)
            # img = np.roll(img ,  size[0]//2- (y+h//2), axis = 0)
            img = img[y-1:y+max(w,h)+1 ,x-1:x+max(w,h)+1]
            cir_img = self.circle_filter(img)
            self.distribute_var.append(np.mean(np.abs(cir_img - img)))
            self.curvature.append(self.curve_fit.fit(cir_img))
                    
        return mean_diameter, min_axis_diff, max_thickness , \
                        self.distribute_var , self.curvature
        
    
    def circle_filter(self , mean_img):
        w, h = mean_img.shape
        n_med = 1.334
    
        circle_filt = cupy.zeros((w,h))
        r = w//2
        center = w/2
        
        x,y = cupy.meshgrid(cupy.arange(w),cupy.arange(h))
        R = cupy.sqrt((x-center)**2+(y-center)**2)
        img_cu = cupy.array(mean_img)
        for rd in range(r):
        
            img_ring = img_cu.copy()
            circle_filt = cupy.zeros((w,h))
            circle_filt[(R >= rd-.5) & (R < rd+.5)] = 1
        
            # img_ring[cupy.equal(circle_filt,0)] = 0
            ring_mean = cupy.mean(img_ring[cupy.equal(circle_filt,1)])
            if ring_mean < n_med:
                pass
            else:            
                img_cu[cupy.equal(circle_filt,1)] = ring_mean
            
        circle = cupy.asnumpy(img_cu)
        return circle
        

class RBC_parameters():
    def __init__(self, path):
        self.round = lambda x : np.round(x , 3)
        self.path = path
        self.n_3d = np.load(path + "\\" + "n_3d.npy")
        self.image_stack = np.load(path + "\\" + "recon_phimap.npy").real
        self.stack_process = stack_process(self.image_stack)
    
    def OV(self ,wl = 0.532):
        ov = []
        for i in self.image_stack:
            ov.append(np.sum(i.real)*(wl/2*np.pi))
        ov_mean = np.mean(np.array(ov))
        return self.round(ov_mean)
    
    def volumn_and_perimeter(self):
        image = self.n_3d
        image[image>0] = 1
        image = image.astype(np.uint8)
        s_area = 0
    
        pre_slice = np.zeros_like(image[:,:,0])
        pre_perimeter = 0
        for i in range(image.shape[0]):
            slices =  image[:,:,i]
            objs_edge,__ = cv2.findContours(slices,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)    # separate different edge
            if len(objs_edge) > 0:
                perimeter = cv2.arcLength(objs_edge[0] , True)
            else:
                perimeter = 0
            area = np.sum(slices)
            intersection_area = np.sum(slices[pre_slice == 1])
            
            s_area = s_area + perimeter + 2* area - intersection_area*2 
            pre_slice = slices
            
        volume = np.sum(image)
        SI = (((6*volume)**(2/3))*(np.pi**(1/3)))/s_area
        
        return self.round(volume) , self.round(s_area) , self.round(SI)
                #volume , surface_area , spericity
            
    def asymmetry_3d(self):
        rot_n_3d = rotate(self.n_3d, 180 , axes=(1,2), reshape=False)
        asymmetry_all = np.sum(np.abs(rot_n_3d - self.n_3d))
        return asymmetry_all
        
    def getall(self):
        ov = self.OV()
        volume, surface_area, sphericity = self.volumn_and_perimeter()
        mean_n = ov/ volume + 1.334  # ov/volumn is the sum of nc - nm(assume its 1.334)
        mean_asy_value = self.asymmetry_3d() / volume  # the average of difference between two side
        
        result = self.stack_process.classifier()
        if len(result) == 3:
            result = list(result) + [0,0]
        mean_diameter, min_axis_diff, max_thickness ,distribute_var ,curvature = result
        
        if curvature == 0:
            curvature = [0,0,0]
        else:
            curvature = np.array(curvature).T
            curvature = [self.round(remove_outlier(x)) for x in curvature]
            distribute_var = self.round(remove_outlier(distribute_var))
            
        print(mean_diameter, min_axis_diff, max_thickness)
        print(distribute_var )
        print(curvature)
        print(ov , volume, surface_area, sphericity , mean_n, mean_asy_value)

        
        return mean_n, ov, mean_diameter, max_thickness, volume, surface_area, \
            sphericity, min_axis_diff, curvature, mean_asy_value

    
path = r"D:\data\2021-03-27\5\phi\0\rbc"
RBC_parameters(path).getall()

    
    
    
    
    
    
    
    