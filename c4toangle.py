# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 17:18:16 2020

@author: YX
"""
import numpy as np
import matplotlib.pyplot as plt


c4 = np.load(r"D:\data\2020-09-17\60x_2\cell2\c4.npy")
deg = np.rad2deg(np.arccos(np.sqrt(c4)))
phimap_stack = np.load(r"D:\data\2020-09-17\60x_2\cell2\phimap_stack.npy",allow_pickle=True)

c4_diff = abs(np.diff(c4))
thres_list = [90.1,180.1,270.1,360.1]
t = 0
plt.plot(np.arange(len(c4)),c4,"o-")
for i in range(0,len(deg),1):
    plt.text(np.arange(len(c4))[i] ,c4[i] , str(i),color = "r",fontsize = 9)
plt.show()

def notturn(point):
    # cal slope
    if c4[point] < 0.8:
        if point == 0 or point == 1 :
            if np.sign(c4[point+1] - c4[point]) == np.sign(c4[point+2] - c4[point+1]):
                return True
            else:
                return False
        elif point == len(c4)-1 or point == len(c4)-2:
            if np.sign(c4[point] - c4[point-1]) == np.sign(c4[point-1] - c4[point-2]):
                return True
            else:
                return False
        else:
            if np.sign(c4[point-1] - c4[point-2]) == np.sign(c4[point+2] - c4[point+1]):
                return True
            else:
                return False
    else:
        return True

for i in range(1,len(deg)):
    
    ori_deg = deg[i]
    thres = thres_list[t]
    if deg[i-1] >  thres:
        t +=1
        thres = thres_list[t]
    
    diff_mean = np.mean(c4_diff[i-1:i+1])
    # print(i,diff_mean)
    
    crossornot = False
    if (deg[i] < 90 and deg[i] > 60) or (deg[i] < 270 and deg[i] > 240):
        if diff_mean*90-abs(deg[i-1] - deg[i]) > 5:
            crossornot = True
    # print(crossornot)
    checkpoint = 0
    while deg[i] < deg[i-1] or crossornot :
        d_list = [abs(90.0001-deg[i]),abs(180.0001-deg[i]),abs(270.0001-deg[i]),abs(360.0001-deg[i])]
        d = d_list[checkpoint]*2
        deg[i] += d
        checkpoint+=1
        crossornot = False
        
        # if (diff_mean < 0.06 and notturn(i))  or notturn(i):
        if diff_mean < 0.06  or notturn(i):
            if deg[i] > thres and deg[i-1] < thres:
                deg[i]-=d
                break
            
    if deg[i] > 360:
        deg[i]-=360
        t=0

    print(i,round(deg[i],3),diff_mean,abs(deg[i-1] - deg[i]),ori_deg,thres)        
    
plt.plot(deg , c4, '.' , c="r")
for i in range(0,len(deg[:35]),1):
    plt.text(deg[i] ,c4[i] , str(i),color = "r",fontsize = 9)
cosline = np.power(np.cos(np.arange(0, 360) * np.pi / 180), 2)
plt.plot(np.arange(0,360),cosline,c = "b")
plt.ylabel("Normalized C4 (a.u.)")
plt.xlabel("Rotation (degree)")
plt.title('c4 vs. $cos^{2}$(θ)')
plt.savefig(r"D:\data\2020-09-17\60x_2\cell2\fit_result")

    