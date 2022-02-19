# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 10:11:55 2021

@author: YX
"""
import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema

normalize = lambda x : (x - np.min(x)) / np.max(x - np.min(x))

for i in range(1,18):
    pth = r"C:\Users\YX\Desktop\temp\8\phi"+"\\"+str(i)+"\\rbc"
    # pth = r"D:\lab\CODE\recon_verify\proj\phase_vari"
            
    elp_para = np.load(pth+ "\\elp_para.npy",allow_pickle=True).T
    phi_stack = np.load(pth + "\\re_crop(autofocus).npy",allow_pickle=True)
    
    shape_list = []
    for i in phi_stack:
        s1 , s2 =  i.shape
        shape_list.append(s1*s2)
        
    short_ax = elp_para[0]   # short axis of all cell
    long_ax = elp_para[1]
    
    # mean_phi = np.mean(elp_para[3])
    pixel_mean_phi = elp_para[3] / np.array(shape_list)
    mean_phi = np.mean(pixel_mean_phi)
    
    std_phi = np.std(pixel_mean_phi)
    remove = np.where((pixel_mean_phi >  (mean_phi + std_phi*2.5)) | (pixel_mean_phi < (mean_phi - std_phi)))[0]
    
    ax_diff = abs(long_ax - short_ax)  # ang correct while large ax len diff
    
    rot_ang = elp_para[2]
    credible = np.array([False]*len(rot_ang))
    credible[ax_diff > 15] = True # if axis difference > 15, its angle is credible
    
    init = 0
    credi_first = find_first = False
    
    front_true = []
    rare_true = []
    for i  in range(1 , len(credible)-1):
        if (credible[i] == True and credible[i-1] == False):
            front_true.append(i)  # first true in each true section
        elif credible[i]== True and credible[i+1] == False:
            rare_true.append(i)  # last true in each true section
            
    # first step confirm whether angle calculation is true on frames which credible is false
    
    for ft in front_true:
        calibrate = True
        while calibrate:
            ft -= 1
            if credible[ft] == True:
                calibrate = False
            elif ft < 0:
                calibrate = False    
            else :
                if abs(rot_ang[ft] - rot_ang[ft+1]) > 45:
                    calibrate = False
                else:
                    credible[ft] = True  # if fron is smooth then change its statement 
    
    
    for rt in rare_true:
        calibrate = True
        while calibrate:
            rt += 1
            if rt >= len(credible):
                break
            if credible[rt] == True:
                calibrate = False
            elif ft == len(rot_ang):
                calibrate = False    
            else :
                if abs(rot_ang[rt] - rot_ang[rt-1]) > 45:
                    calibrate = False
                else:
                    credible[rt] = True  # if fron is smooth then change its statement 
    
    # second step calibrate still false frame
    find_true = True
    for c, ( ang , credi) in enumerate(zip(rot_ang , credible)) :
        if credi == True:
            find_true = True
            continue
        else:
            f_true = rot_ang[c-1]
            c_ = c
            while find_true and c_ < len(rot_ang):  # find front and rare True value
                if credible[c_] == True:
                    r_true = rot_ang[c_]
                    find_true = False
                    break
                c_ += 1
            
            rot_ang[c] = [ang , 180-ang][np.argmin([abs(ang - rot_ang[c-1]) , abs(180 - ang - rot_ang[c-1])])]
    
    
    #step 3 correct angle with large variation
    ang_diff =np.diff(rot_ang)
    ang_diff = np.insert(ang_diff , 0 , 0)
    turn_pos = [] 
    for c , (ang , d_ang , d_axis) in enumerate(zip( rot_ang , ang_diff , ax_diff)):
        # print(c , (ang , d_ang , d_axis) )
        if abs(d_ang) < 90 and abs(d_ang) > 10 and d_axis < 10:
            f_mean = []
            c_ = c
            while c_-1 >= 0 :
                if abs(ang_diff[c_-1]) < 10:
                    f_mean.append(rot_ang[c_-1])
                if len(f_mean) >= 4:
                    f_mean = np.mean(f_mean)
                    break
                c_ -= 1
        
            r_mean = []
            c_ = c
            while c_+1 < len(rot_ang) :
                if abs(ang_diff[c_+1]) < 10:
                    r_mean.append(rot_ang[c_+1])
                if len(r_mean) >= 4:
                    r_mean = np.mean(r_mean)
                    break
                c_ += 1
            f_mean = np.mean(f_mean)
            r_mean = np.mean(r_mean)
            rot_ang[c] = (f_mean + r_mean)//2
        elif abs(d_ang) > 90 :
            if abs(rot_ang[c] - rot_ang[c-1]) > 90:
                turn_pos.append(c)
                if len(turn_pos) > 1 and turn_pos[-1] - turn_pos[-2] < 20:
                    turn_pos.pop(-1)
                    rot_ang[c] = 180 - ang
                    # print(c , ang , rot_ang[c])
                
    #%%
    #step 4 turn to rotate back angle
    ang_diff =np.diff(rot_ang)
    ang_diff = np.insert(ang_diff , 0 , 0)
    frame = []
    rotate_ang = []
    turn_t = -1
    
    # new_rotang = []
        
    for c , (ang , d_ang , d_axis) in enumerate(zip(rot_ang , ang_diff , ax_diff)):
        # print(ang)
        if abs(d_ang) < 10 or abs(d_ang) > 90:
            # new_rotang.append(ang)
            frame.append(c)
            if abs(d_ang) > 90 and (rot_ang[c-1] < 30 or rot_ang[c-1] > 150) :
                turn_t += 1
                if turn_t == 0:
                    direction = [1,-1][d_ang > 0] # find out its clock or counterclock 
            
            if turn_t < 0:
                ang = ang 
            else:
                if direction == -1: # counter clock
                    ang = -180*(turn_t +1) + ang
                else:
                    ang = 180*(turn_t +1) + ang
            # ang = [ang , ang-180][(turn_t % 2) != 0]
           
            rotate_ang.append(ang)
        
    for c , i in enumerate(remove):
        if i in frame:
            frame.pop(c)
            rotate_ang.pop(c)
        
        
    for c , (ori , ang) in enumerate(zip(frame , rotate_ang)):
        print(c ,ori ,  ang)
        
    from scipy.ndimage import rotate
    
    phistack = np.load(pth + "\\re_crop(autofocus).npy",allow_pickle=True)
    new_phistack = []
    for i , ang in zip(frame , rotate_ang):
        recrop_phi = phistack[i]
        recrop_phi.real = rotate(recrop_phi.real,ang,reshape=False,mode="constant")
        recrop_phi.imag = rotate(recrop_phi.imag,ang,reshape=False,mode="constant")
        # plt.imshow(recrop_phi.real)      
        # plt.show()    
        new_phistack.append(recrop_phi)
        
    # np.save(pth + "\\framAFTrotate.npy" , frame)
    # np.save(pth + "\\frame.npy" , frame)
    np.save(pth + "\\rotate_phi.npy" , np.array(new_phistack,dtype=object))
    
        


    