# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:21:22 2021

@author: YX
"""
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import pandas as pd
from matplotlib.patches import Rectangle
from time import time
from tqdm import tqdm
import json
from glob import glob
import os
import warnings
warnings.filterwarnings("ignore")

nonneg = lambda a : np.array([[x, np.inf][x<0] for x in a])

class cell:
    def __init__(self , name):
        self.name = name
        self.frame = []
        self.x_all = []
        self.y_all = []
        self.diff_x = 0
        self.diff_y = 0
        self.get_new = True
        self.work = True
        self.bg = 0
        
    def __str__(self):
        return "name : {0} , first frame : {1}, average y : {2}, x range : {3}~{4}"\
    .format(self.name,self.frame[0], np.mean(self.y_all), self.x_all[0], self.x_all[-1])
    __repr__ = __str__   # with repr = str, class can be print in terminal without print
    
    def update(self ,f , x ,y):
        self.x_all.append(x)
        self.y_all.append(y)
        self.frame.append(f)
        self.diff_x = np.nan_to_num(np.mean(np.abs(np.diff(self.x_all))))
        self.diff_y = np.nan_to_num(np.mean(np.diff(self.y_all)))
        self.get_new = True
        if x >= 3000:
            self.work = False
    def estimate_pos(self):
        # print(self.name)
        if self.x_all[-1] + self.diff_x >= 3072:
            self.work = False
        else:
            self.x_all.append(self.x_all[-1] + self.diff_x)
            self.y_all.append(self.y_all[-1]+ self.diff_y)
            self.get_new = True
    def get_prev(self):
        # if self.x_all[-1] == 10000:
        #     return 0
        # else:
            # print(self.get_new)
        if self.get_new == False:
            self.estimate_pos()
        self.get_new = False
        return [self.name , self.x_all[-1] , self.y_all[-1]]
        
        
file_p = r"C:\Users\YH\Desktop\temp\4\analysis"
# all_file_p = r"E:\2021-04-08\4"
t1 = time()
kernel = np.ones((10,10), np.uint8)
# mask_stack = np.load(file_p+r'\mask_stack.npy')
# df = pd.DataFrame(columns=["number" , "fig" , "paras"])  #"ctr_x" , "ctr_y", "w" , "h"
# TBD = {}
c_num = 0
prev_df = np.array([])
cells = []

#%%
all_mask = []
mask_sum = np.zeros((3072,3072))
for c, i in tqdm(enumerate(sorted(glob(file_p+"/*.npy"), key = os.path.getmtime))):
    if c<5000:
        frame_mask = []
        i = np.load(i)
        i = cv2.morphologyEx(i,cv2.MORPH_OPEN, kernel).astype(np.uint8)
        i = cv2.resize(i , (3072,3072),cv2.INTER_NEAREST)
        
        mask_sum += i
        
        if c>0 and (c+1)%10 == 0:
            # print("\n+++++++++++" + str(c))
            mask_sum[mask_sum <= 1] = 0
            mask_sum[mask_sum > 1] = 1
            
            # plt.imshow(mask_sum)
            # plt.title(str(c))
            # plt.show()
            
            contours, hierarchy = cv2.findContours(mask_sum.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            mask_sum = np.zeros((3072,3072))   
            bbox_ctr = [(cv2.boundingRect(cnt) , cv2.moments(cnt)) for cnt in contours]
            # FOR every bbox(center) in figure
           
            TBD = [] 
            if len(bbox_ctr) == 0:
                continue
            else:
                for (x,y,w,h) , M in bbox_ctr:
                    # print((x,y,w,h))
                    if max(w , h) > 220 or min(w , h) < 50:
                        continue        # means two cell combined
                    
                    else:
                        # print(x ,y)
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # print(cX , cY ,w,h)            
                        # add all bbox to dataframe in first figure
                        if len(prev_df) == 0 or len(prev_df[0]) == 0 :
                            # obj_name = "rbc"+str(c_num+1)
                            CELL = cell(c_num)
                            CELL.update(c , cX , cY)
                            cells.append(CELL)
                            c_num+=1
                        
                        # calculate x and y distance compare to all bbox in previous figure
                        else :        
                            x_dist = nonneg(cX - prev_x)
                            y_dist = cY - prev_y
                            dist = np.sqrt(x_dist**2 + y_dist**2)

                            # print(x_dist )
                            
                            count = 0
                            for i in np.argsort(dist):   # find which is avaliable to add
                                if dist[i] < 200 and cells[i].get_new == False:
                                    cells[prev_name[i]].update(c , cX , cY)
                                    break
                                count += 1
                            # print(count , len(dist))
                            if count == len(dist) and cX < 100:  #　only if new cell x < 50 create otherwise neglect
                                CELL = cell(c_num)
                                CELL.update(c , cX , cY)
                                cells.append(CELL)
                                c_num+=1
                            elif count == len(dist) and len(dist) == len(prev_x):
                                if abs(y_dist[np.argsort(dist)[0]]) < 50 and cells[prev_name[np.argsort(dist)[0]]].get_new == False:
                                    cells[prev_name[np.argsort(dist)[0]]].update(c , cX , cY)
                
                
                # prev_df = np.array([c_p.get_prev() for c_p in cells if c_p.get_prev() != 0]).T
                prev_df = []
                for c_p in cells:
                    # prev = c_p.get_prev()
                    if c_p.work:
                        prev_df.append(c_p.get_prev())
                prev_df = np.array(prev_df).T
                
                if len(prev_df) == 0:  #incase at the middle the last one cell over 3000 and be stop adding   
                    continue
                else:
                    prev_name , prev_x, prev_y = prev_df
                    prev_name = prev_name.astype(int)
                    prev_x = prev_x.astype(float).astype(int)
                    prev_y = prev_y.astype(float).astype(int)
                # print(prev_y)
                # print("+++++++++++")
#%% calculate bg
# [name frame , x , y , bg , size]
cell_dict = {}

for c , i in enumerate(cells):
    y_mean = int(np.mean(i.y_all))  
    length = len(i.y_all) 
    if length > 15 :
        x_diff = int(np.mean(np.diff(i.x_all))) 
        if x_diff > 10:
            name = str(i.frame[0]) +"_" +str(i.y_all[0])
            init_f = i.frame[0]
    
            cell_dict.update({name : [init_f ,init_f+length , y_mean ,i.x_all , i.y_all ]})

    
cell_paras = np.array(list(cell_dict.values()))
cell_name = list(cell_dict.keys())

start_all = cell_paras.T[0]
end_all = cell_paras.T[1]
y_all = cell_paras.T[2]


for c , cp in enumerate(cell_paras):
    start_f  , end_f , y , _ , _ = cp
    end_f += 5
    ## find front single bg
    find_bg = True
    while find_bg:
        possible_ovl = np.where(( start_all < end_f) & (end_all > end_f))[0]
        if len(possible_ovl) == 0:
            bg = end_f
            find_bg = False
        else:
            for i in possible_ovl:
                if y_all[i] - y < 100:
                    end_f = end_all[i]
                    break 
                else :
                    bg = end_f
                    find_bg = False
                    break
    
    cell_dict[cell_name[c]].append(bg)
    #[start_frame , end_frame , y_mean , all x , all y , bg]
    
    
    # for i in range(-(len(cell_paras)-c)+1 , c):
    #     if i > 0:
    #         bg = end_f+1
            
    #         if 
#%%
import pickle
with open(file_p+r"\position_data.pkl" ,"wb") as pd_file:
    pickle.dump(cell_dict , pd_file)
    pd_file.close()
    
#%%
# cell_dict = {}

# for i in cells:
#     if len(i.x_all) > 15:
#         name = str(i.frame[0]) +"_" +str(i.y_all[0])
#         cell_dict.update({name : [i.x_all[0] , i.y_all[0] , })
#         print(i.name)
#         print(i.y_all[0])
#         print(i.frame[0])
#         print("+++++++++++")
    


