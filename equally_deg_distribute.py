# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:08:34 2020

@author: YX
"""

import numpy as np
import matplotlib.pyplot as plt

path = r"D:\data\2020-10-17\rbc2\phi\rbc1"
def saverecon(deg,newphimaplist,c4,save = True):
    if save :
        np.save(path+"\\recon_deg.npy", deg)
        np.save(path+"\\recon_phimap.npy", newphimaplist)
        np.save(path + "\\c4.npy",c4)

deg = np.load(r"D:\data\2020-10-17\rbc2\phi\rbc1\recon_deg.npy")
phimap  = np.load(r"D:\data\2020-10-17\rbc2\phi\rbc1\recon_phimap.npy")
c4  = np.load(r"D:\data\2020-10-17\rbc2\phi\rbc1\c4.npy")

combine = [(phimap[i],deg[i],c4[i]) for i in range(len(deg))]

combine = sorted(combine, key = lambda x : x[1])

phimap , deg ,c4 = zip(*combine)

deg_diff = np.diff(deg)
# print(np.median(deg_diff))

new_deg = [deg[0]]
new_c4 = [c4[0]]
new_phi = [phimap[0]]
diff_sum = 0

for i in range(1 , len(deg)):
    diff_sum += deg_diff[i-1]
    if diff_sum > 4:
        new_c4.append(c4[i])
        new_deg.append(deg[i])
        new_phi.append(phimap[i])
        diff_sum = 0
cosline = np.power(np.cos(np.arange(0, 360) * np.pi / 180), 2)
plt.plot(new_deg ,new_c4 , "o" , c="r")
plt.plot(np.arange(0,360),cosline,c = "b")
plt.ylabel("Normalized C4 (a.u.)")
plt.xlabel("Rotation (degree)")
plt.title('c4 vs. $cos^{2}$(θ)')
plt.savefig(r"D:\data\2020-09-17\60x_2\cell2\fit_result")
    
saverecon(new_deg,new_phi,new_c4)
    
    
    
    
    

