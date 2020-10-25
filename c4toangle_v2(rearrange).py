# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:17:25 2020

@author: YX
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import rotate
import cv2

path = r"D:\data\2020-10-17\rbc2\phi\rbc1"
c4 = np.load(path + "\\c4.npy",allow_pickle=True)
phimap = np.load(path +"\\phimap_stack.npy",allow_pickle=True)
phi = np.load(path + "\\phi.npy",allow_pickle=True)

def saverecon(deg,newphimaplist,c4,save = True):
    if save :
        np.save(path+"\\recon_deg.npy", deg)
        np.save(path+"\\recon_phimap.npy", newphimaplist)
        np.save(path + "\\c4.npy",c4)

# c4 = np.loadtxt(r"D:\data\2020-09-17\60x_2\cell1\c4.txt")
# c4 = np.loadtxt(r"D:\LOAD.txt")

#find turn point
c4_s = savgol_filter(c4,7,3)  #smooth c4 term list
candidate = []
turn_point =  []
minormax = False # true :find min ; false :find max
start_point = minormax  # start at 0(True) or 90(False) degree

for i in range(1,len(c4_s)-1):
    
    if c4[i] == 1:
        turn_point.append((c4_s[i],i)+(minormax,))
        minormax = not(minormax)
        candidate = []
        continue
    elif c4[i] == 0:
        turn_point.append((c4_s[i],i)+(minormax,))
        minormax = not(minormax)
        candidate = []
        continue
    
    get_turnpoint = False
    if minormax == True:
        if c4_s[i] < c4_s[i-1] and c4_s[i] < c4_s[i+1]:
            candidate.append((c4_s[i],i)) 
            if c4_s[i] < 0.5:
                get_turnpoint = True
                
    elif minormax == False:
        if c4_s[i] > c4_s[i-1] and c4_s[i] > c4_s[i+1]:
            candidate.append((c4_s[i],i))
            if c4_s[i] > 0.5:
                get_turnpoint = True

    print(i  , minormax , candidate)
           
    if (abs(c4_s[i] - np.mean([c[0] for c in candidate])) > 0.1 and get_turnpoint) or i == len(c4_s)-1:
    # if get_turnpoint or i == len(c4_s)-1:
        if minormax == True :
            turn_point.append(min(candidate)+(minormax,))
        elif minormax == False : 
            turn_point.append(max(candidate)+(minormax,))
        # print(candidate)
        candidate = []
        minormax = not(minormax)
print(turn_point)

plt.plot(c4 , "o-")
# plt.plot(c4_s , "o-")
for i in range(0,len(c4_s),1):
    plt.text(i ,c4_s[i] , str(i),color = "g",fontsize = 8)
plt.show()
plt.show()

#merge c4 and phimap and phi for sorting
phi_c4 = [(phimap[i],c4[i],phi[i]) for i in range(len(c4))] #merge phimap and c4

sort_phi_c4 = []
for i in range(len(turn_point)+1):
    ub = turn_point[i][1] if i != len(turn_point) else len(c4)
    lb = [turn_point[i-1][1] , 0][i == 0]  #[F,T]
    sort_method = turn_point[i][2] if i!=len(turn_point) else not(turn_point[i-1][2])
    sortpart = sorted(phi_c4[lb:ub] ,key = lambda x : x[1], reverse=sort_method)
    sort_phi_c4+=sortpart

turn_index = [turn_point[i][1] for i in range(len(turn_point))]

#unpack and calculate c4 to degree
newphimaplist , newc4list , newphilist = zip(*sort_phi_c4)
deg = np.rad2deg(np.arccos(np.sqrt(newc4list)))
deg[0] = [deg[0]+90,deg[0]][start_point]
for i in range(0,len(newc4list),1):
    plt.text(i ,newc4list[i] , str(i),color = "r",fontsize = 8)
plt.plot(newc4list , "o-")
plt.show()
for i in range(0,len(newphilist),1):
    plt.text(i ,newphilist[i] , str(i),color = "r",fontsize = 8)
plt.plot(newphilist , "o-")
plt.show()


thres_list = [90.1,180.1,270.1,360.1]
t = 0
for i in range(1,len(deg)):
    ori_deg = deg[i]
    checkpoint = 0
    while deg[i] < deg[i-1]:
        d_list = [abs(90.0001-deg[i]),abs(180.0001-deg[i]),abs(270.0001-deg[i]),abs(360.0001-deg[i])]
        d = d_list[checkpoint]*2
        deg[i] += d
        checkpoint+=1
                    
    if deg[i] > 360:
        deg[i]-=360
        t=0
    # print(i,c4[i] , deg[i])
  

#unify size of each frame
sizelist = []
newphimaplist = list(newphimaplist)
for i in newphimaplist:
    sizelist.append(i.shape[0])
max_size = max(sizelist)
for count , i in enumerate(newphimaplist):
    i_shape = i.shape[0]
    size_diff = max_size - i_shape
    # print(count)
    # resize_real = cv2.resize(i.real , dsize =(max_size,max_size),interpolation=cv2.INTER_CUBIC)
    # resize_imag = cv2.resize(i.imag , dsize =(max_size,max_size),interpolation=cv2.INTER_CUBIC)
    resize_real = np.pad(i.real,(size_diff//2,size_diff//2),"constant",constant_values = (0,0))
    resize_imag = np.pad(i.imag,(size_diff//2,size_diff//2),"constant",constant_values = (0,0))
    newphimaplist[count] = resize_real + 1j*resize_imag
    
    #retrieve phi degree rotation 
    # newphimaplist[count].real = rotate(newphimaplist[count].real,newphilist[count],reshape=False,mode="constant")
    # newphimaplist[count].imag= rotate(newphimaplist[count].imag,newphilist[count],reshape=False,mode="constant")

    
saverecon(deg,newphimaplist,c4)

    
for i in range(0,len(deg),10):
    plt.text(deg[i] ,c4[i] , str(i),color = "r",fontsize = 9)
cosline = np.power(np.cos(np.arange(0, 360) * np.pi / 180), 2)
plt.plot(deg ,c4 , "o" , c="r")
plt.plot(np.arange(0,360),cosline,c = "b")
plt.ylabel("Normalized C4 (a.u.)")
plt.xlabel("Rotation (degree)")
plt.title('c4 vs. $cos^{2}$(θ)')
plt.savefig(r"D:\data\2020-09-17\60x_2\cell2\fit_result")
    

    
    
    