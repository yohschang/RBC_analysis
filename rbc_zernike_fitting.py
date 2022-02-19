'''
check out  interferometer_zernike
'''

import numpy as np
import os
import multiprocessing as mp
from glob import glob
import matplotlib.pyplot as plt
from matplotlib import cm
# import rbc_shape_curvefit
# from TVmin import TvMin
import cupy
from scipy.signal import medfilt 
from scipy.signal import argrelextrema,savgol_filter
from scipy.ndimage import gaussian_filter

normalization = lambda x : (x -np.min(x))/ np.max(x -np.min(x))
def f_r_zero(x):
    x = np.append(x,0)
    return np.insert(x ,0 ,0) 

#%%
class Coefficient(object):

    __coefficients__ = []
    __zernikelist__ = [ "Z00 Piston or Bias",
						"Z11 x Tilt",
						"Z11 y Tilt",
						"Z20 Defocus",
						"Z22 Primary Astigmatism at 45",
						"Z22 Primary Astigmatism at 0",
						"Z31 Primary y Coma",
						"Z31 Primary x Coma",
						"Z33 y Trefoil",
						"Z33 x Trefoil",
						"Z40 Primary Spherical",
						"Z42 Secondary Astigmatism at 0",
						"Z42 Secondary Astigmatism at 45",
						"Z44 x Tetrafoil",
						"Z44 y Tetrafoil",
						"Z51 Secondary x Coma",
						"Z51 Secondary y Coma",
						"Z53 Secondary x Trefoil",
						"Z53 Secondary y Trefoil",
						"Z55 x Pentafoil",
						"Z55 y Pentafoil",
						"Z60 Secondary Spherical",
						"Z62 Tertiary Astigmatism at 45",
						"Z62 Tertiary Astigmatism at 0",
						"Z64 Secondary x Trefoil",
						"Z64 Secondary y Trefoil",
						"Z66 Hexafoil Y",
						"Z66 Hexafoil X",
						"Z71 Tertiary y Coma",
						"Z71 Tertiary x Coma",
						"Z73 Tertiary y Trefoil",
						"Z73 Tertiary x Trefoil",
						"Z75 Secondary Pentafoil Y",
						"Z75 Secondary Pentafoil X",
						"Z77 Heptafoil Y",
						"Z77 Heptafoil X",
						"Z80 Tertiary Spherical"]

    def __init__(self,
			Z1=0, Z2=0, Z3=0, Z4=0, Z5=0, Z6=0, Z7=0, \
			Z8=0, Z9=0, Z10=0, Z11=0, Z12=0, Z13=0, Z14=0, \
				 Z15=0, Z16=0, Z17=0, Z18=0, Z19=0, Z20=0, Z21=0, \
				 Z22=0, Z23=0, Z24=0, Z25=0, Z26=0, Z27=0, Z28=0, \
				 Z29=0, Z30=0, Z31=0, Z32=0, Z33=0, Z34=0, Z35=0, Z36=0, Z37=0):
        if type(Z1) == list:
            self.__coefficients__ = Z1 + [0]*(37-len(Z1))
        else:
            self.__coefficients__ = [Z1, Z2, Z3, Z4, Z5, Z6, Z7,
					Z8, Z9, Z10, Z11, Z12, Z13, Z14, Z15, Z16, Z17,
					Z18, Z19, Z20, Z21, Z22, Z23, Z24, Z25, Z26,
					Z27, Z28, Z29, Z30, Z31, Z32, Z33, Z34, Z35, Z36, Z37]
    def outputcoefficient(self):
	    return self.__coefficients__
    def listcoefficient(self):
        m = 0
        label1 = ""
        label2 = ""
        for i in self.__coefficients__:
            if i != 0:
                print('Z'+str(m+1)+' = ',i,self.__zernikelist__[m])
                label1 = label1 + 'Z'+str(m+1)+' = '+str(i)+"\n"
                label2 = label2 + 'Z'+str(m+1)+' = '+str(i)+"  "
            m = m + 1
        return [label1,label2]

    def zernikelist(self):
        m = 1
        for i in self.__zernikelist__:
            print("Z"+str(m)+":"+i)
            m = m + 1

    def zernikemap(self, label = True):
        theta = np.linspace(0, 2*np.pi, 400)
        rho = np.linspace(0, 1, 400)
        [u,r] = np.meshgrid(theta,rho)
        X = r*np.cos(u)
        Y = r*np.sin(u)
        Z = __zernikepolar__(self.__coefficients__,r,u)
        fig = plt.figure(figsize=(12, 8), dpi=80)
        ax = fig.gca()
        # im = plt.pcolormesh(X, Y, Z, cmap=cm.RdYlGn)
        im = plt.pcolormesh(X, Y, Z, cmap="jet")

        if label == True:
            plt.title('Zernike Polynomials Surface Heat Map',fontsize=18)
            ax.set_xlabel(self.listcoefficient()[1],fontsize=18)
        plt.colorbar()
        ax.set_aspect('equal', 'datalim')
        plt.show()
        return Z

    
def __zernikepolar__(coefficient,r,u):

    Z = [0]+coefficient
    Z1  =  Z[1]  * 1*(np.cos(u)**2+np.sin(u)**2)
    Z2  =  Z[2]  * 2*r*np.cos(u)
    Z3  =  Z[3]  * 2*r*np.sin(u)        
    Z4  =  Z[4]  * np.sqrt(3)*(2*r**2-1)   #DEFOCUS
    Z5  =  Z[5]  * np.sqrt(6)*r**2*np.sin(2*u)   # OBLIQUE PRIMARY ASTIGMATISM
    Z6  =  Z[6]  * np.sqrt(6)*r**2*np.cos(2*u)  # hor primary astigmatism
    Z7  =  Z[7]  * np.sqrt(8)*(3*r**2-2*r)*np.sin(u)   # vertical coma
    Z8  =  Z[8]  * np.sqrt(8)*(3*r**2-2*r)*np.cos(u)   # horizontal coma
    Z9  =  Z[9]  * np.sqrt(8)*r**3*np.sin(3*u)    # 
    Z10 =  Z[10] * np.sqrt(8)*r**3*np.cos(3*u)   # 
    Z11 =  Z[11] * np.sqrt(5)*(6*r**4-6*r**2 + 1)  # primary spherical abberation
    Z12 =  Z[12] * np.sqrt(10)*(4*r**2-3)*r**2*np.cos(2*u)  #  ver / hor. sec astigmatism
    Z13 =  Z[13] * np.sqrt(10)*(4*r**2-3)*r**2*np.sin(2*u) # oblique sec astigmatism
    Z14 =  Z[14] * np.sqrt(10)*r**4*np.cos(4*u)
    Z15 =  Z[15] * np.sqrt(10)*r**4*np.sin(4*u)
    Z16 =  Z[16] * np.sqrt(12)*(10*r**4-12*r**2+3)*r*np.cos(u)
    Z17 =  Z[17] * np.sqrt(12)*(10*r**4-12*r**2+3)*r*np.sin(u)
    Z18 =  Z[18] * np.sqrt(12)*(5*r**2-4)*r**3*np.cos(3*u)
    Z19 =  Z[19] * np.sqrt(12)*(5*r**2-4)*r**3*np.sin(3*u)
    Z20 =  Z[20] * np.sqrt(12)*r**5*np.cos(5*u)
    Z21 =  Z[21] * np.sqrt(12)*r**5*np.sin(5*u)
    Z22 =  Z[22] * np.sqrt(7)*(20*r**6-30*r**4+12*r**2-1)
    Z23 =  Z[23] * np.sqrt(14)*(15*r**4-20*r**2+6)*r**2*np.sin(2*u)
    Z24 =  Z[24] * np.sqrt(14)*(15*r**4-20*r**2+6)*r**2*np.cos(2*u)
    Z25 =  Z[25] * np.sqrt(14)*(6*r**2-5)*r**4*np.sin(4*u)
    Z26 =  Z[26] * np.sqrt(14)*(6*r**2-5)*r**4*np.cos(4*u)
    Z27 =  Z[27] * np.sqrt(14)*r**6*np.sin(6*u)
    Z28 =  Z[28] * np.sqrt(14)*r**6*np.cos(6*u)
    Z29 =  Z[29] * 4*(35*r**6-60*r**4+30*r**2-4)*r*np.sin(u)
    Z30 =  Z[30] * 4*(35*r**6-60*r**4+30*r**2-4)*r*np.cos(u)
    Z31 =  Z[31] * 4*(21*r**4-30*r**2+10)*r**3*np.sin(3*u)
    Z32 =  Z[32] * 4*(21*r**4-30*r**2+10)*r**3*np.cos(3*u)
    Z33 =  Z[33] * 4*(7*r**2-6)*r**5*np.sin(5*u)
    Z34 =  Z[34] * 4*(7*r**2-6)*r**5*np.cos(5*u)
    Z35 =  Z[35] * 4*r**7*np.sin(7*u)
    Z36 =  Z[36] * 4*r**7*np.cos(7*u)
    Z37 =  Z[37] * 3*(70*r**8-140*r**6+90*r**4-20*r**2+1)


    Z = Z1 + Z2 +  Z3+  Z4+  Z5+  Z6+  Z7+  Z8+  Z9+ \
		Z10+ Z11+ Z12+ Z13+ Z14+ Z15+ Z16+ Z17+ Z18+ Z19+ \
		Z20+ Z21+ Z22+ Z23+ Z24+ Z25+ Z26+ Z27+ Z28+ Z29+ \
		Z30+ Z31+ Z32+ Z33+ Z34+ Z35+ Z36+ Z37
    return Z

def fitting(Z,n):
    fitlist = []
    l = len(Z)
    x2 = np.linspace(-1, 1, l)
    y2 = np.linspace(-1, 1, l)
    [X2,Y2] = np.meshgrid(x2,y2)
    r = np.sqrt(X2**2 + Y2**2)  # turn cartesian coordinate to polar coordinate
    u = np.arctan2(Y2, X2)    # turn cartesian coordinate to polar coordinate
    for q in range(n):
        if q not in [3,5,10,11]:
            fitlist.append(0)
        else :
            C = [0]*q+[1]+[0]*(37-q-1)   # coeff of i'th term become 1 
            ZF = __zernikepolar__(C, r, u)
            Z_tot = Z * ZF
            Z_tot[r > 1] = 0
            a = sum(sum(Z_tot))*2*2/l/l/np.pi   # why this can be seen as coef??
            fitlist.append(np.around(a,3))

    l1 = len(fitlist)
    fitlist = fitlist+[0]*(37-l1)
    Z_new = __zernikepolar__(fitlist,r,u)
    for i in range(l):
        for j in range(l):
            if x2[i]**2+y2[j]**2>1:
                Z_new[i][j]=0
    # C = Coefficient(fitlist)  #output zernike Coefficient class
    # C.zernikemap()
    return fitlist , Z_new

# from scipy.ndimage import rotate , zoom
# path = r"256_0.npy"
# phasemap = np.load(path).real
# # phasemap = zoom(phasemap,1)
# phasemap = (phasemap - np.min(phasemap))/np.max((phasemap - np.min(phasemap)))

# el_ang = 75
# phasemap = rotate(phasemap,el_ang,reshape=False,mode="constant")

# plt.figure(dpi = 300)
# plt.imshow(phasemap,cmap = "jet",vmin = 0 , vmax = 3)
# plt.show()
# Z1,C = fitting(np.fliplr(phasemap.real),5)

#%%

class RBC_fitting():
    def __init__(self , path , isdash = False , save = False , mode = 1):
        self.path = path
        self.isdash = isdash
        self.save = save
        self.theta = np.array([])
        self.phi = np.array([])
        self.phistack = np.array([])
        self.mode = mode

    def dhash_row_col(self , image, size=8):
        width = size + 1
        grays = np.resize(image,(width,width))
        row_hash = 0
        col_hash = 0
        for y in range(size):
            for x in range(size):
                row_bit = grays[y,x] < grays[y,x+1]
                row_hash = row_hash << 1 | row_bit
                col_bit = grays[y,x] < grays[y+1,x]
                col_hash = col_hash << 1 | col_bit
        return (row_hash, col_hash)

    def dhash_int(self, image, size=16):
        row_hash, col_hash = self.dhash_row_col(image, size=size)
        return row_hash << (size * size) | col_hash

    def get_num_bits_different(self , hash1, hash2):
        return bin(hash1 ^ hash2).count('1')


    def zernikeFitting(self, rimap, isdash):
        # new_map = cupy.array(rimap.copy())
        ZC ,C = fitting(rimap, 13)
        hash1 = self.dhash_int(rimap)
        hash2 = self.dhash_int(C)
        DIST = self.get_num_bits_different(hash1, hash2)
        if isdash:
            if DIST > 18:
                DIST = 100

        # z_fit = opticspy_kernel.zernike.Coefficient(C)
        # ideal = C.zernikemap(label = False) #plot
        return ZC, DIST

    def normalize(self,x,x_min):
        x = np.asarray(x)
        return (x - x.min()) / (np.max(x - x.min()))
    
    def read_rbc(self ,dash = False):

        fitting_data = []
        if self.mode == 1:
            phi_stack = np.load( self.path + "\\rotate_phi.npy",allow_pickle=True)
        else:
            phi_stack = np.load( self.path + "\\rotate_phi_lite.npy",allow_pickle=True)

        self.phistack = phi_stack
        print(self.phistack.shape) 
        # phi_stack = np.load(self.path + "\\with_ideal.npy",allow_pickle=True)
        # ideal = rbc_shape_curvefit.get_ideal_90(self.path + "\\re_crop(autofocus).npy",1)
        for i in phi_stack:
            # tv = TvMin(cupy.array(i) , lamb = 0.15 , iteration = 50)
            # tv.minimize()
            # i = cupy.asnumpy(tv.getResultImage())
            # print(i.shape)
            # i = (i - np.min(i))/np.max(i - np.min(i))
            # i = normalization(i)
            fitting_data.append((i.real,dash))
        return fitting_data,np.array(phi_stack)

    def remove_outer(self , c4):
        
        med_c4 = medfilt(c4)
        std = np.std(np.abs(c4-med_c4))
        differ = []
        differ_t_f = np.zeros_like(c4)
        differ_t_f[c4>med_c4+1.5*std] = 1
        differ_t_f[c4<med_c4-1.5*std] = 1
        
        for i in range(len(c4)):
            if i == 0:
                differ.append(np.abs(c4[i] - c4[i+1]))
            elif i == len(c4)-1:
                differ.append(np.abs(c4[i] - c4[i-1]))
            else:
                differ.append(np.abs(c4[i] - c4[i+1])+np.abs(c4[i] - c4[i-1]))
        for i in range(1,len(differ)):
            if differ[i] > differ[i-1] and differ[i] > differ[i+1]:
                if (c4[i] - c4[i-1] > 0 and c4[i+1] - c4[i] > 0) or (c4[i] - c4[i-1] < 0 and c4[i+1] - c4[i] < 0):
                    pass
                else:
                    differ_t_f[i] = 1
                    # print(theta[i-1] , theta[i] , theta[i+1])
            else :
                pass
        return np.array(differ_t_f)
    
    def section_find_ang(self , upright , flatten, C4 , C6):
        t_point = sorted(list(upright) + list(flatten))
        # t_point = t_point[:-1]
        frame_num = np.array([])
        degs = np.array([])
        
        c46 = C4+C6
        save_c46 = []
        # plt.plot(c46)
        # for c , i in enumerate(range(len(c46))):
        #     col = "b"
        #     s = 30
        #     if c in upright:
        #         col = "r"
        #         s = 60
        #     elif c in flatten:
        #         col = "orange"
        #         s = 60
        #     plt.scatter(c, c46[c],s = s,c = col)
        # plt.title('C$_{4+5}$')
        # plt.ylabel("C$_{4+5}$ value")
        # plt.xlabel("frame")
        # plt.show()
        plt.plot(c46, "o-")
        plt.title("c46")
        plt.show()
        
        ori = ['d' , 'u'][c46[0] < c46[t_point[0]]]  #cell rotation orientation decide start angle 
        
        if ori == 'd':
            deg_shift = [0,-180,180,-360]*10
        else:
            deg_shift = [-180,180,-360,0]*10
        
        init = 0
        
        
        for c , (i , d_init) in enumerate(zip(t_point , deg_shift)):
            if i == len(c46)-1:
                break
            
            frame = np.arange(init , i+1)
            if c == 0:
                section = (c46[init:i+1] + (0-c46[i]))
            else:
                section = c46[init:i+1]
            
            # plt.figure(dpi = 300)
            # for cc , mm in enumerate(range(len(section))):
            #     cc+=init
            #     col = "b"
            #     s = 30
            #     if cc in upright:
            #         col = "r"
            #         s = 60
            #     elif cc in flatten:
            #         col = "orange"
            #         s = 60
            #     plt.scatter(cc, section[mm],s = s,c = col)
            # plt.plot(np.arange(len(section),dtype = int)+int(init) , section)
            # plt.title('section of C$_{4+5}$')
            # plt.ylabel("C$_{4+5}$ value")
            # plt.xticks(range(0,5))
            # plt.xlabel("frame")
            # plt.show()

                
            if c46[i+1] > c46[init]:
                n_section = np.delete(section , section < c46[init])
                frame = np.delete(frame , section < c46[init])
            else :
                n_section = np.delete(section , section > c46[init])
                frame = np.delete(frame , section > c46[init])
            if c > 0:
                n_section = normalization(n_section)
            deg = np.abs(np.rad2deg(np.arccos(np.sqrt(n_section))) + d_init)
            # print(deg)
            
            degs = np.append(degs , deg[1:])
            frame_num = np.append(frame_num , frame[1:])
            save_c46.append(n_section[1:])
        
            init = i
            
        
        # for ff in frame_num:
        #     save_c46.append(c46[int(ff)])     

        self.phistack = self.size_uniform(frame_num)
        
        if self.mode ==1:
            np.save(self.path + "\\recon_deg.npy", degs)
            np.save(self.path + "\\recon_phimap.npy", self.phistack)
            np.save(self.path + "\\c46.npy", save_c46)
        else :
            np.save(self.path + "\\recon_deg_lite.npy", degs)
            np.save(self.path + "\\recon_phimap_lite.npy", self.phistack)
        # return frame_num , degs
    
    def size_uniform(self, frame_num):
        sizelist = []
        frame_num = frame_num.astype(int)
        for i in frame_num :
            sizelist.append(self.phistack[i].real.shape[0])
        max_size = max(sizelist)
        
        newphimaplist = np.zeros((len(frame_num), max_size , max_size))
        # newphimaplist = np.zeros((len(frame_num), max_size , max_size) , dtype = np.complex128)
        for count, i in enumerate(frame_num):
            img = self.phistack[i]
            i_shape = img.shape[0]
            size_diff = max_size - i_shape
            resize_real = np.pad(img.real, (size_diff // 2, size_diff // 2), "constant", constant_values=(0, 0))
            resize_imag = np.pad(img.imag, (size_diff // 2, size_diff // 2), "constant", constant_values=(0, 0))
            newphimaplist[count] = resize_real + 1j * resize_imag
        return newphimaplist
        

    def run(self):
            # print(__name__)
            # path = [r"D:\data\2020-08-13(rotate_rbc)\60x\20\cell1\42.npy"]#,r"D:\data\2020-08-13(rotate_rbc)\60x\20\cell1\115.npy"]
            fitting_data , phimap_stack = self.read_rbc(self.isdash)
            pool = mp.Pool(6)
            zc_list = [pool.starmap(self.zernikeFitting,fitting_data)]
            pool.close()
            pool.join()

            zc = []
            ideal_fit = []
            for i in np.array(zc_list)[:,:,0][0]:
                zc.append(i)
            for j in np.array(zc_list)[:,:,1][0]:
                ideal_fit.append(j)

            # zc = np.array(zc)[np.array(ideal_fit) != 100 ]
            # ideal_fit = np.array(ideal_fit)[np.array(ideal_fit) != 100 ]
            # phimap_stack = phimap_stack[np.array(ideal_fit) != 100 ]

            # c1 = list(np.array(zc)[:, 1])
            # c2 = list(np.array(zc)[:, 2])
            c4 = list(np.array(zc)[:, 3])
            c6 = list(np.array(zc)[:, 5])
            c12 = list(np.array(zc)[:, 11])
            c11 = list(np.array(zc)[:, 10])

            c6 , c12 = normalization(np.array(c6)), normalization(np.array(c12))    # for upright
            c4  , c11 = normalization(np.array(c4)), normalization(np.array(c11))
            c_411 = c4 + np.abs(1-c11)

            c411_f = gaussian_filter(c_411,2) 
            c12_f = gaussian_filter(c12,2) 
            # if len(np.where(c12 == np.max(c12))[0]) > 1:
            #     c12[np.where(c12 == np.max(c12))[0]] += np.random.rand()*0.1
            # c12_f = c12
            
            print("get coefficient")
            c12_f = f_r_zero(c12_f)
            c411_f = f_r_zero(c411_f)

            upright_idx = argrelextrema(c12_f, np.greater, order = 15,mode = "clip")[0]-1
            print(upright_idx)
            # upright = c12[upright_idx]
            # print(upright)
            # upright_idx = upright_idx[upright > 0.6]
            for c , i in enumerate(upright_idx):
                rmin = [0 , i-15][i-15 >= 0] 
                rmax = [len(c12) , i+16][i+16 <= len(c12)]
                upright_idx[c] = np.argmax(c12[rmin:rmax]) + rmin
            upright = c12[upright_idx]
            upright_idx = upright_idx[upright > 0.6]
            print("upright : " + str(upright_idx))

            plt.plot(c12_f)
            plt.plot(c12)
            # for c , i in enumerate(range(len(c12))):
            #     col = "b"
            #     s = 30
            #     if c in upright_idx:
            #         col = "r"
            #         s = 60
            #     plt.scatter(i, c12[i],s = s,c = col)
            plt.scatter(upright_idx, c12[upright_idx], s = 4,c = "r", marker = "+")
            plt.title('C$_{12}$')
            plt.ylabel("C$_{12}$ value")
            plt.xlabel("frame")
            plt.show()
            
            mean_d = np.mean(np.diff(upright_idx))
            if np.isnan(mean_d):
                mean_d = len(c411_f) // 3
            print("mean_d" + str(mean_d))
            
            flatten_idx = argrelextrema(c411_f, np.greater, order =int(mean_d*0.5),mode = "clip")[0]-1


            for c , i in enumerate(flatten_idx):
                rmin = [0 , i-15][i-15 >= 0] 
                rmax = [len(c_411) , i+16][i+16 <= len(c_411)]
                flatten_idx[c] = np.argmax(c_411[rmin:rmax]) + rmin
            flatten = c_411[flatten_idx]
            flatten_idx = flatten_idx[flatten > 1.3]
            print("flatten" + str(flatten_idx))


            plt.plot(c_411)
            # plt.plot(c411_f)
            # for c , i in enumerate(range(len(c_411))):
            #     col = "b"
            #     s = 30
            #     if c in flatten_idx:
            #         col = "orange"
            #         s = 60
                # plt.scatter(i, c_411[i],s = s,c = col)
            plt.scatter(flatten_idx, c_411[flatten_idx],c = "r" )
            plt.title('C$_{5+13}$')
            plt.ylabel("C$_{5+13}$ value")
            plt.xlabel("frame")
            plt.show()
             

            
            

            
            self.section_find_ang(upright_idx , flatten_idx, c4, c6)
            
            return 0
            
            ''' origin filter
            # c1,c2 = c1[:-1],c2[:-1]
            ideal_fit = np.array(ideal_fit[:-1])
            
            np.save(self.path + "\\c4_2", c4)
            
            differ = self.remove_outer(c4)
            c4 = np.array(c4)[differ == False]
            c6 = c6[differ == False]
            c12 = c12[differ == False]
            # phi = np.arctan2(c1, c2) * 180 / np.pi
            ideal_fit = ideal_fit[differ == False]
            self.phistack = phimap_stack[differ == False]       
            self.theta = self.normalize(c_sum,c_sum.min())
            
            # self.theta = c4
            plt.figure(dpi = 300)
            plt.plot(self.theta , "o-")
            plt.show()
            
            self.save_data()
            '''

    def save_data(self):
        if self.save:
            # self.phi -= 90
            # self.phi[self.phi < 0] = self.phi[self.phi < 0]+360
            # output = [(phimap_stack[i] , theta[i]) for i in range(len(theta))]
            np.save(self.path + "\\c4", self.theta)
            # np.save(self.path + "\\phi", self.phi)
            np.save(self.path + "\\phimap_stack.npy", self.phistack)

        else:
            pass
        
if __name__ == "__main__":
    
    path = r"E:\data\2021-03-27\5\phi\0"
    # filefolder = glob(path+"/*")
    # for f in filefolder:
    fit = RBC_fitting(path +"/rbc" ,save =True, mode = 1)
    fit.run()
        # for i in range(0,2):
            # fit = RBC_fitting(r"D:\data\2020-12-06\6\phi\rbc"+str(i) ,save =False)
        # fit = RBC_fitting(r'D:\data\2021-04-08\4\phi'+ '\\'+str(j)+'\\rbc' ,save =False, mode = 1)

