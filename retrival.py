import sys
sys.path.append(r"D:\lab\CODE\phase retrival")
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
import scipy
import scipy.stats
import time
import os
from scipy.ndimage import gaussian_filter1d,median_filter
import glob
from unwrap import unwrap
import cv2
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import pickle
isneg = lambda x : [x , 0][x < 0]


class phase_retrieval:
    def __init__(self , path , para_dict , title, number ,bg_p,pad = False, save = True , plot = True ,flip =True, unwrap_method = 1):
        self.path = path
        self.number = number
        self.bg_p = bg_p
        
        self.s_frame , _ , _ , self.x , self.y , self.bg = para_dict
        self.x , self.y = np.array(self.x) , np.array(self.y)
        
        self.save = save
        self.plot = plot
        self.unwrap_method = unwrap_method
        self.pad = pad
        self.title = title
        self.flip = flip      
        self.side_l = 200
        
        self.mean_phi = []

    def imgft(self ,s,i,j,bg_or_sp):     # x = 0 for bg x = 1 for sp
        row = s.shape[0]
        col = s.shape[1]
        out = np.fft.fft2(s)
        shift = np.fft.fftshift(out)
        shift1 = shift.copy()

        shift1[round((1*row)*2/5):row,0:col] = 0.01 #set a crop regeion approximatly to find max position
        if bg_or_sp == "bg":
            place = np.where(shift1 == np.max(shift1))  #find max position cor
            # print(place)
            newcenter = shift[int(place[0])-row//8:int(place[0])+row//8,int(place[1])-col//8:int(place[1])+col//8] #use max as center to crop a region
            center_pos = [place[0], place[1]]

        elif bg_or_sp == "sp":
            newcenter = shift[int(i) - row // 8 :int(i) + row // 8 ,int(j) - col // 8 :int(j) + col // 8 ]
            center_pos = [0,0]
        
        if self.pad :
            shrink_r = 6  # 1:2048(3072);  ; 2:1024 ; 4:512...
                          # 1: 3072 ; 6:1024...
            pad_size_c = (row//2-newcenter.shape[0])//shrink_r
            pad_size_r = (col//2-newcenter.shape[1])//shrink_r
            # print(pad_size_c,pad_size_r)
            inv_fshift = np.pad(newcenter,((pad_size_c,pad_size_c) , (pad_size_r,pad_size_r)))
            # print(newcenter.shape , inv_fshift.shape)
        else:
            inv_fshift = newcenter  # reverse fourier

        img_amp = np.fft.ifft2(inv_fshift)**2  #calculate amplitude
        img_recon = np.arctan2(np.imag(np.fft.ifft2(inv_fshift)),np.real(np.fft.ifft2(inv_fshift)))   #calculate phase difference as wrapped image
        return img_recon,img_amp, center_pos,np.fft.ifft2(inv_fshift)

    def crop(self, x,y , side_l):
        x0 = [0 , x-side_l-20][x-side_l-20 >= 0]
        x1 = [3072 , x+side_l+20][x+side_l+20 < 3072]
        y0 = [0 , y-side_l][y-side_l >= 0]
        y1 = [3072 , y+side_l][y+side_l < 3072]
        
        return list(map(int,[y0 , y1 , x0 , x1]))
    
    def pos_cali(self , img):
        proj = np.sum(img , axis = 0)
        ctr_pos = len(proj)*2//3
        cell_pos = np.mean(np.where(proj > 30)[0])
        dist_diff = cell_pos - ctr_pos
        return int(np.nan_to_num(dist_diff))

    def run(self):
        count = 0
        path = self.path
        if not os.path.exists(path+"\\phi"):
            os.makedirs(path+"\\phi")
        if not os.path.exists(path+"\\phi"+"\\"+str(self.number)):
            os.makedirs(path+"\\phi"+"\\"+str(self.number))
            
        bgpath = self.path+"\\"+str(self.title)+"_"+str(self.bg)+".bmp"
        bg_img = cv2.imread(bgpath,0)
        # bg_img = cv2.imread(self.bg_p,0)
            
        frame = self.s_frame
        print(self.y[0])
        x_diff = np.insert(np.diff(self.x),0,0)
        y_diff = np.insert(np.diff(self.y),0,0)
        dist_diff = 0
        for c ,(x , y) in enumerate(zip(self.x , self.y)):
            for i in range(9):
                # try : 
                save_path = self.path +"\\phi"+"\\"+str(self.number)+"\\"+str(frame)
                
                sppath = self.path+"\\"+str(self.title)+"_"+str(frame)+".bmp"
                sp = cv2.imread(sppath,0)
                frame += 1
                   
                region = self.crop(x,y  , 200 )
              
                sp = sp[region[0]:region[1] , region[2]:region[3]]

                bg = bg_img[region[0]:region[1] , region[2]:region[3]]
                # plt.imshow(bg , cmap = "gray")
                # plt.show()
                if c == len(x_diff):
                    break
                # x = x + int(x_diff[c])//9
                # y = y + int(y_diff[c])//9

                f_bg, a_bg, p1 ,new_bg= self.imgft(bg,0,0,"bg")    #find bg center first then directory used as sp img crop center after FT
                f_sp, a_sp, p2 , new_sp = self.imgft(sp,p1[0],p1[1],"sp")    #find bg center first then directory used as sp img crop center after FT
    
                # f_result= f_sp-f_bg
                f_result= -f_sp+f_bg
            
                a_result = np.sqrt(a_sp/a_bg)  # calculate amplitude
    
                output1 = unwrap(f_result)
                
                while np.mean(output1) > 3:
                    output1 -= np.pi
                while np.mean(output1) < -3:
                    output1 += np.pi
                
                comb = output1+1j*(a_result.real)
            

                if self.flip:
                    comb = np.fliplr(comb)
                    # print(comb.shape)
                    output1 = np.fliplr(output1)
    
                if self.save:
                    if np.sum(comb.real[comb.real > 0]) > 1000 and np.sum(comb.real[comb.real > 0]) < 5000:
                        np.save(save_path,comb)
                    # np.save(path + "\\phi"+"\\"+str(i),comb)
    
                if self.plot:
                    plt.imshow(comb.real,cmap='jet',vmin = 0)#,vmax = 3)
                    plt.colorbar()
                    plt.title(str(frame-1) + "  " + str(x))
                    # plt.title(str())
                    plt.show()
                # except :
                #     pass
                self.mean_phi.append(np.sum(comb.real[comb.real > 0]))
                # print(np.mean(self.mean_phi))
                if len(self.mean_phi) >= 8 and np.mean(self.mean_phi) < 500:
                        return 0
            dist_diff = self.pos_cali(output1)
            self.x+=(dist_diff) -10
            # print(x)
            # print(self.x[:5])

def check_dup(df):
    all_x = df[3]
    all_y = df[4]
    
    diff_y = np.diff(all_y)
    count = 0
    while True:
        if diff_y[count] != 0:
            break
        count +=1
    if count == 0:
        return df 
    else :
        gap_y , gap_x = (all_y[count+1] - all_y[0])/count , (all_x[count+1] - all_x[0])/count
        all_y[1:count+1] = np.arange(1,count+1)*gap_y + all_y[0] 
        all_x[1:count+1] = np.arange(1,count+1)*gap_x + all_x[0] 
        df[3] = all_x
        df[4] = all_y
        return df            
            
if __name__ == "__main__":
    
    pos_data_f = r'C:\Users\YH\Desktop\temp\4\analysis'
    bg_p = r"C:\Users\YH\Desktop\temp\1_bg.bmp"
    with open(pos_data_f+r"\position_data.pkl" ,"rb") as f:
        pos_data = pickle.load(f)
    # pos_data['1999_290'][-1] = 1985
    # pos_data['649_2929'][3][:11] = np.linspace(54,798,12)[:-1]
    del_p = []
    for c,  i in tqdm(enumerate(pos_data)):
        if c>0:
            paras = check_dup(pos_data[i])
            #[start_frame , end_frame , y_mean , all x , all y , bg]
            title = int(pos_data_f.split("\\")[-2])
            phase_retri = phase_retrieval 
            path = "\\".join(pos_data_f.split("\\")[:-1])
            try:
                rt = phase_retri(path , paras ,title , c ,bg_p,pad = True, plot = False,save = True,flip =False).run()
            except ValueError:
                # print("error")
                rt = 0
            if rt is not None:
                del_p.append(c)
        # (path , para_dict , title, number ,pad = False, save = True , plot = True ,flip =True, unwrap_method = 1)
    print("del : " + str(del_p))