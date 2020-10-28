# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:57:00 2020

@author: YX
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import rotate
import cv2

class angle_calculation():
    def __init__(self , uniform_angle = False , save = False , plot = False):
        self.phimap = np.array([])
        self.c4 = np.array([])
        self.phi = np.array([])
        self.path = None
        self.recon_deg = np.array([])
        self.recon_phimap = np.array([])
        self.ud_c4 = np.array([])   #c4 after uniform distribute

        self.uni_distribute = uniform_angle
        self.saveornot = save
        self.plotornot = plot

    def run(self,path):
        self.c4 = np.load(path + "\\c4.npy", allow_pickle=True)
        self.phimap = np.load(path + "\\phimap_stack.npy", allow_pickle=True)
        self.phi = np.load(path + "\\phi.npy", allow_pickle=True)

        self.find_turn_point(plot_pos = False)
        self.plotc4_deg(self.recon_deg , self.c4 , plot = self.plotornot)
        self.uniform_distribute(self.recon_phimap, self.recon_deg, self.c4, sample_size=4, re_distribute=self.uni_distribute)
        if self.uni_distribute:
            self.plotc4_deg(self.recon_deg , self.ud_c4 , plot = self.plotornot)
        self.saverecon(path, self.recon_deg, self.recon_phimap, self.c4, save= self.saveornot)


    def find_turn_point(self , plot_pos = False):
        c4_s = savgol_filter(self.c4,7,3)  #smooth c4 term list
        candidate = []
        turn_point =  []
        minormax = False # true :find min ; false :find max
        start_point = minormax  # start at 0(True) or 90(False) degree

        for i in range(1,len(c4_s)-1):

            if self.c4[i] == 1:
                turn_point.append((c4_s[i],i)+(minormax,))
                minormax = not(minormax)
                candidate = []
                continue
            elif self.c4[i] == 0:
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

            if (abs(c4_s[i] - np.mean([c[0] for c in candidate])) > 0.1 and get_turnpoint) or i == len(c4_s)-1:
            # if get_turnpoint or i == len(c4_s)-1:
                if minormax == True :
                    turn_point.append(min(candidate)+(minormax,))
                elif minormax == False :
                    turn_point.append(max(candidate)+(minormax,))
                # print(candidate)
                candidate = []
                minormax = not(minormax)

        # plot the position and mark the number of c4 and c4_s
        if plot_pos:
            plt.plot(self.c4, "o-")
            plt.plot(c4_s , "o-")
            for i in range(0, len(c4_s), 1):
                plt.text(i, c4_s[i], str(i), color="g", fontsize=8)
            plt.show()

        # merge phimap and c4 and sort by c4
        phi_c4 = [(self.phimap[i], self.c4[i], self.phi[i]) for i in range(len(self.c4))]  # merge phimap and c4
        sort_phi_c4 = []
        for i in range(len(turn_point) + 1):
            ub = turn_point[i][1] if i != len(turn_point) else len(self.c4)
            lb = [turn_point[i - 1][1], 0][i == 0]  # [F,T]
            sort_method = turn_point[i][2] if i != len(turn_point) else not (turn_point[i - 1][2])
            sortpart = sorted(phi_c4[lb:ub], key=lambda x: x[1], reverse=sort_method)
            sort_phi_c4 += sortpart

        turn_index = [turn_point[i][1] for i in range(len(turn_point))]

        # unpack and calculate c4 to degree
        newphimaplist, newc4list, newphilist = zip(*sort_phi_c4)
        deg = np.rad2deg(np.arccos(np.sqrt(newc4list)))
        deg[0] = [deg[0] + 90, deg[0]][start_point]

        if plot_pos:
            for i in range(0, len(newc4list), 1):
                plt.text(i, newc4list[i], str(i), color="r", fontsize=8)
            plt.plot(newc4list, "o-")
            plt.show()

        #turn c4 to degree
        thres_list = [90.1, 180.1, 270.1, 360.1]
        t = 0
        for i in range(1, len(deg)):
            ori_deg = deg[i]
            checkpoint = 0
            while deg[i] < deg[i - 1]:
                d_list = [abs(90.0001 - deg[i]), abs(180.0001 - deg[i]), abs(270.0001 - deg[i]), abs(360.0001 - deg[i])]
                d = d_list[checkpoint] * 2
                deg[i] += d
                checkpoint += 1

            if deg[i] > 360:
                deg[i] -= 360
                t = 0


        # unify size of each frame
        sizelist = []
        newphimaplist = list(newphimaplist)
        for i in newphimaplist:
            sizelist.append(i.shape[0])
        max_size = max(sizelist)
        for count, i in enumerate(newphimaplist):
            i_shape = i.shape[0]
            size_diff = max_size - i_shape
            resize_real = np.pad(i.real, (size_diff // 2, size_diff // 2), "constant", constant_values=(0, 0))
            resize_imag = np.pad(i.imag, (size_diff // 2, size_diff // 2), "constant", constant_values=(0, 0))
            newphimaplist[count] = resize_real + 1j * resize_imag
        self.c4 = newc4list
        self.recon_deg = deg
        self.recon_phimap = newphimaplist

    def uniform_distribute(self, phimap, deg, c4, sample_size = 4, re_distribute = True):
        if re_distribute:
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
                if diff_sum >= sample_size:
                    new_c4.append(c4[i])
                    new_deg.append(deg[i])
                    new_phi.append(phimap[i])
                    diff_sum = 0
            self.recon_deg = new_deg
            self.recon_phimap = new_phi
            self.ud_c4 = new_c4
        else:
            pass

    def plotc4_deg(self,deg , c4 , plot = True):
        if plot:
            cosline = np.power(np.cos(np.arange(0, 360) * np.pi / 180), 2)
            plt.plot(deg ,c4 , "o" , c="r")
            plt.plot(np.arange(0,360),cosline,c = "b")
            plt.ylabel("Normalized C4 (a.u.)")
            plt.xlabel("Rotation (degree)")
            plt.title('c4 vs. $cos^{2}$(θ)')
            plt.show()
        else:
            pass

    def saverecon(self , path, deg, newphimaplist, c4, save = True):
        if save :
            np.save(path+"\\recon_deg.npy", deg)
            np.save(path+"\\recon_phimap.npy", newphimaplist)
            # np.save(path + "\\c4.npy",c4)
        else:
            pass


if __name__ == "__main__" :
    anglecal = angle_calculation(uniform_angle=False , plot = True , save = True)
    anglecal.run(r"D:\data\2020-10-17\rbc2\phi\rbc1")

