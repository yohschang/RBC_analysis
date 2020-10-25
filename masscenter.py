import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import skimage.morphology as sm
import os
from scipy.optimize import curve_fit
import pandas as pd
import pickle
from scipy.ndimage import rotate

def region_resize(x,y,w,h,region_adj):
    x -= region_adj;    y -= region_adj;    w += 2*region_adj;    h += 2*region_adj
    if x < 0 : x = 0
    if x > phase_map_size : x = phase_map_size 
    if y < 0  : y = 0 
    if y > phase_map_size : y = phase_map_size 
    if y+h > phase_map_size : h = phase_map_size-y 
    if x+w > phase_map_size : w = phase_map_size-x  
    return x,y,w,h

def pad_output(crop_obj_rimap, y_mass , x_mass , outputsize):
    opsize = outputsize//2
    crop_obj_rimap = np.pad(crop_obj_rimap,((opsize,opsize),(opsize,opsize)),"constant")
    x_mass = x_mass+opsize
    y_mass = y_mass+opsize
    crop_obj_rimap = crop_obj_rimap[x_mass-opsize:x_mass+opsize,y_mass-opsize:y_mass+opsize]  # crop object with image center being coordinate of center of mass

    return crop_obj_rimap

def analyze_type(rbc_db,frame,ctms):    #define which rbc it belong
    # print(frame-1)
    distance = []
    # pre_ctms = (rbc_db.loc[rbc_db["frame"] == frame-1, ["mass_center","type"]]["mass_center"]).values.tolist()   # ctms = mass center
    # types = (rbc_db.loc[rbc_db["frame"] == frame-1, ["mass_center","type"]]["type"]).values.tolist()    # ctms = mass center
    # types = list(map(int,[x[3:] for x in types]))
    pre_ctms = []
    str_types = []

    for i in position:
        prct  = position[i]
        dist = np.sqrt((ctms[0]-prct[0])**2 + (ctms[1]-prct[1])**2)  #calculate new ctms dist to each rbc in previous frame 
        # dist = (ctms[1]-prct[1]) #calculate new ctms dist to each rbc in previous frame 
        distance.append(dist)
        pre_ctms.append(prct)
        str_types.append(i)
    int_types = list(map(int,[x[3:] for x in str_types]))
        
    distance.append(999)  #prevent there is only one element in list
    near_ctms = int(np.where(distance == np.min(distance))[0])
    if ctms[0] < pre_ctms[near_ctms][0] or distance[near_ctms]<10:
        if abs(ctms[1] - pre_ctms[near_ctms][1])>5:
            print("y move large")
            ctms[1] = pre_ctms[near_ctms][1]   # if y jump larger than 5 -> caliberate to previous 
        name = "rbc"+str(int_types[near_ctms])
        position[name] = ctms
    else :
        name = "rbc"+str(max(int_types)+1)
        position.update({name:ctms})
    return name , ctms

def find_ellipse_angle(contour):    # calculate the angle in order to rotate back to verticle
    ellipse = cv2.fitEllipse(contour)
    centers ,radius  ,angle = ellipse
    cx,cy = int(round(centers[0])),int(round(centers[1]))
    ax1,ax2 =  int(round(radius[0])),int(round(radius[1]))
    center = (cx,cy)
    axes = (ax1,ax2)
    # print(center ,radius  ,angle)
    return center,axes,round(angle) 
        

def find_masscenter(phase_img,ori_rimap,region_adj,rbc_db,frame,real_frame):
    val1 , phase_img = cv2.threshold(phase_img,0,255,cv2.THRESH_OTSU)   #otsu binary phase img
    # val1 , phase_img = cv2.threshold(phase_img,int(np.max(phase_img)*0.45),255,cv2.THRESH_BINARY)#cv2.THRESH_OTSU)   #otsu binary phase img
    # print(val1)
    phase_img = sk.filters.median(sm.closing(phase_img,sm.disk(3)))
   # result = cv2.morphologyEx(phase_img,cv2.MORPH_CLOSE,kernel=(5,5),iterations=10)

    # plt.imshow(phase_img)
    # plt.show()
    # ori_rimap[phase_img == 0] = 0
    
    # result = cv2.morphologyEx(phase_img,cv2.MORPH_CLOSE,kernel=(5,5),iterations=10)
    # blur_phase_img = cv2.GaussianBlur(phase_img, (5,5), 0) 
    # phase_img_edge = cv2.Canny(phase_img, threshold1=250, threshold2=250)   #find edge
    objs_edge,__ = cv2.findContours(phase_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    # separate different edge
    
    collect = False
    centermass = {}
    count =0
    if len(objs_edge) != 0:
        for count , o in enumerate(objs_edge):
            [row,column] = np.meshgrid(np.arange(phase_img.shape[1]), np.arange(phase_img.shape[0]))
            
            # perimeter = cv2.arcLength(o, True)
            area = cv2.contourArea(o)
            const_points = len(o)
            
            #### remove objet at edge after autofocus ###
            if const_points > 3 :
                x, y, w, h = cv2.boundingRect(o)     # output each object region
                x ,y , w , h = region_resize(x, y, w, h, region_adj)
                crop_obj_rimap = np.zeros_like(phase_img)
                crop_obj_rimap[y:y+h,x:x+w] = ori_rimap[y:y+h,x:x+w]
                optical_volumn = area * np.sum(np.abs(crop_obj_rimap))
                crop_size = max(w,h)
                
                # plt.figure(count)
                # plt.imshow(ori_rimap[y:y+h,x:x+w],cmap="jet",vmax=3.5,vmin=0.5)
                # plt.colorbar()
                # plt.show()
                
                # print("......................................")
                # print(x,y)
                # print("const_points"+str(const_points))
                # print("optical_volumn"+str(optical_volumn))
                # print("average" + str(np.average(crop_obj_rimap[y:y+h,x:x+w])))
                if np.average(crop_obj_rimap[y:y+h,x:x+w]) > 0.3  and  optical_volumn > 130000:
                    if len(o) >= 5:
                        el_center , el_radius , el_ang = find_ellipse_angle(o)
                    else:
                        el_ang = 0
                    
           
                    # cv2.ellipse(phase_img, el_center , el_radius , el_ang,0,360 ,(0,255,0), 10) 
                    # print(el_center , el_radius, el_ang)
                    # plt.imshow(ori_rimap.real,cmap = "jet",vmin = 0)
                    # plt.show()
                    # plt.imshow(phase_img)
                    # plt.show()
                    # print(x,y)
                    collect = True
                    # np.save(path + "\\" + str(count) + str([y,x]), ori_rimap[x:x+h,y:y+w])
                    # print(sum(crop_obj_rimap[phase_img == 255]*row[phase_img == 255]),sum(crop_obj_rimap[phase_img == 255]))
                    x_ct_mass = int(round(sum(crop_obj_rimap[phase_img == 255]*row[phase_img == 255])/sum(crop_obj_rimap[phase_img == 255]))) #calculate x coodinate of center of mass
                    y_ct_mass = int(round(sum(crop_obj_rimap[phase_img == 255]*column[phase_img == 255])/sum(crop_obj_rimap[phase_img == 255]))) #calculate y coodinate of center of mass
                    
                    r_x_ctms = x_ct_mass+window_x  # the mass center position in origin figure
                                       
                    if frame == 1:
                        rbc_type = "rbc"+str(count+1)
                        position.update({rbc_type:[r_x_ctms,y_ct_mass]})
                    else:
                        rbc_type,[r_x_ctms,y_ct_mass] = analyze_type(rbc_db,frame,[r_x_ctms,y_ct_mass])
                    
                    cal_rimap = ori_rimap.copy()
                    crop_obj_rimap = pad_output(cal_rimap,r_x_ctms-window_x,y_ct_mass,crop_size)

                    crop_obj_rimap.real = rotate(crop_obj_rimap.real,el_ang,reshape=False,mode="constant")
                    crop_obj_rimap.imag = rotate(crop_obj_rimap.imag,el_ang,reshape=False,mode="constant")

                    series = pd.Series({"frame":frame,"mass_center":[r_x_ctms,y_ct_mass],"rbc_array":crop_obj_rimap,"type":rbc_type,"real_frame":real_frame},name = str(count))
                    rbc_db = rbc_db.append(series)
                    print(r_x_ctms,y_ct_mass)
                    plt.figure(count)
                    plt.title(str(real_frame))
                    plt.imshow(crop_obj_rimap.real,cmap="jet",vmax=3.5,vmin =0 )
                    plt.colorbar()
                    plt.show()
    if collect :  
        frame += 1
    return rbc_db,frame,r_x_ctms


position = {}
frame = 1
rbc_db = pd.DataFrame(columns=["frame","mass_center","rbc_array","type","real_frame"])

for count , rf in enumerate(range(315,535,1)): 
    print(rf)
    path = r"D:\data\2020-10-17\rbc2\phi\rbc1"
    # path = r"D:\data\2020-07-22\Bead\1\SP\common\2\data"#save dir
    # if not os.path.isdir(path):
    #     os.mkdir(path)
    phase_map_size = 100
    ori_rimap_path = path +"\\"+str(rf)+".npy"#np array phasemap dir
    ori_rimap = np.load(ori_rimap_path)
    # phase_img = np.round((np.abs(ori_rimap.real)+0.5)**8,0).astype(np.uint8)     #load phase retrival output bmp
    phase_img = np.round(((ori_rimap.real)+0.5)**9).astype(np.uint8)     #load phase retrival output bmp
    phase_img[phase_img<0] = 0
    
    # plt.imshow(phase_img,cmap = "jet")
    # plt.colorbar()
    # plt.show()


    region_adj = 7 # peripheral length
    
    window_y = 0
    if count == 0 or ct_x+50>512:
        window_x= 412
    elif ct_x - 50 > 0 :
        window_x = ct_x - 50 


    # plt.figure(1234)
    # plt.imshow(ori_rimap.real,cmap = "jet",vmax=3.5,vmin =0)
    #  plt.imshow(ori_rimap.real,cmap = "jet",vmax=3.5,vmin =0)
    # plt.show() 

    rbc_db , frame ,ct_x  = find_masscenter(phase_img[window_y:window_y+100,window_x:window_x+100],ori_rimap[window_y:window_y+100,window_x:window_x+100],
                                                    region_adj,rbc_db,frame,rf)
    # rbc_db , frame = find_masscenter(phase_img[95:150,0:100],ori_rimap[95:150,0:100],region_adj,rbc_db,frame,rf)
    # print(rf)
#%%  save data frame
type_list = list(set(rbc_db["type"].values.tolist()))
rbc_db = rbc_db.set_index('type')
for idx , IDX in enumerate(type_list):
    savefile = rbc_db.loc[IDX].to_pickle(path+"//"+IDX+".pickle")

        
    

# parsed = json.loads(result)
# json.dumps(parsed, indent=4)  


# plt.show()