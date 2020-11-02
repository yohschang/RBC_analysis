import numpy as np
import os
import multiprocessing as mp
import glob
import matplotlib.pyplot as plt
import rbc_zernike_fit

#%% calculate dash value
class RBC_fitting():
    def __init__(self , path):
        self.path = path
        self.isdash = False
        self.theta = np.array([])
        self.phi = np.array([])
        self.phistack = np.array([])

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


    #%% zernike fitting
    def zernikeFitting(self, rimap, isdash):
        # new_map = cupy.array(rimap.copy())

        ZC ,C = rbc_zernike_fit.fitting(rimap, 5)
        hash1 = self.dhash_int(rimap)
        hash2 = self.dhash_int(C)
        DIST = self.get_num_bits_different(hash1, hash2)
        if isdash:
            if DIST > 18:
                DIST = 100

        # z_fit = opticspy_kernel.zernike.Coefficient(C)
        # ideal = C.zernikemap(label = False) #plot
        return ZC, DIST

    def normalize(self,x):
        x = np.asarray(x)
        return (x - x.min()) / (np.ptp(x))

    def read_rbc(self , filepath,dash = False):

        fitting_data = []
        phi_stack = np.load( self.path + "\\re_crop.npy",allow_pickle=True)
        for i in phi_stack:
            fitting_data.append((i.real,dash))
        return fitting_data,np.array(phi_stack)

    def remove_outer(self , c4):
        differ = []
        for i in range(len(c4)):
            if i == 0:
                differ.append(np.abs(c4[i] - c4[i+1]))
            elif i == len(c4)-1:
                differ.append(np.abs(c4[i] - c4[i-1]))
            else:
                differ.append(np.abs(c4[i] - c4[i+1])+np.abs(c4[i] - c4[i-1]))
        differ_t_f = [False]
        for i in range(1,len(differ)):
            if differ[i] > differ[i-1] and differ[i] > differ[i+1]:
                if (c4[i] - c4[i-1] > 0 and c4[i+1] - c4[i] > 0) or (c4[i] - c4[i-1] < 0 and c4[i+1] - c4[i] < 0):
                    differ_t_f.append(False)
                else:
                    differ_t_f.append(True)
                    # print(theta[i-1] , theta[i] , theta[i+1])
            else :
                differ_t_f.append(False)
        return np.array(differ_t_f)

    def run(self):

        if __name__ == "__main__":
            path = r"D:\data\2020-10-17\rbc2\phi\rbc1"
            # path = [r"D:\data\2020-08-13(rotate_rbc)\60x\20\cell1\42.npy"]#,r"D:\data\2020-08-13(rotate_rbc)\60x\20\cell1\115.npy"]
            fitting_data , phimap_stack = self.read_rbc(path , self.isdash)
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

            zc = np.array(zc)[np.array(ideal_fit) != 100 ]
            ideal_fit = np.array(ideal_fit)[np.array(ideal_fit) != 100 ]
            phimap_stack = phimap_stack[np.array(ideal_fit) != 100 ]

            c1 = list(np.array(zc)[:, 1])
            c2 = list(np.array(zc)[:, 2])
            c4 = list(np.array(zc)[:, 3])

            differ = self.remove_outer(c4)
            c4 = np.array(c4)[differ == False]
            phi = np.arctan2(c1, c2) * 180 / np.pi
            ideal_fit = ideal_fit[differ == False]
            self.phistack = phimap_stack[differ == False]
            self.phi = phi[differ == False]

            theta = self.normalize(c4)


    def save_data(self , save = False ):
        if save:
            # self.phi -= 90
            # self.phi[self.phi < 0] = self.phi[self.phi < 0]+360
            # output = [(phimap_stack[i] , theta[i]) for i in range(len(theta))]
            np.save(self.path + "\\c4", self.theta)
            # np.save(self.path + "\\phi", self.phi)
            np.save(self.path + "\\phimap_stack.npy", self.phimap_stack)

        else:
            pass

fit = RBC_fitting(r"D:\data\2020-10-17\rbc2\phi\rbc1")
fit.run()

        # theta = c4
        # print(theta)





        # new_theta = [theta[0]]
        # for i in range(1, len(  theta) - 1):
        #     filt = np.average([theta[i - 1], theta[i], theta[i + 1]])
        #     new_theta.append(filt)
        # new_theta.append(theta[-1])
        # filter_remove = np.abs(new_theta - theta)
        # fr_mean = np.mean(filter_remove)
        # fr_std = np.std(filter_remove)
        # theta = theta[filter_remove < fr_mean+fr_std]
        # ideal_fit = ideal_fit[filter_remove < fr_mean+fr_std]
        # new_theta = np.array(new_theta)[filter_remove < fr_mean+fr_std]


        # # plt.plot(new_theta)
        # for count , f in enumerate(ideal_fit):
        #     thres = [5,14,18,24]
        #     if f <= thres[0] :
        #         color = "r"
        #     elif f<=thres[1] and f>thres[0]:
        #         color = "b"
        #     elif f <= thres[2] and f > thres[1]:
        #         color = "g"
        #     elif f <= thres[3] and f > thres[2]:
        #         color = "y"
        #     else :
        #         color = "k"
        #     plt.figure(1)
        #     # plt.scatter(np.arange(len(theta)), theta, c='r', marker='x')
        #     plt.scatter(count, theta[count], c=color, marker='x')
        #     for i in range(count):
        #         if differ[i] == True:
        #             C = "r"
        #         else:
        #             C = "b"
        #         plt.text(i ,theta[i] ,str(i+1),color = C)
        #     # tx = np.arange(count)
        #     # plt.text(count, theta[count] , tx)
        #
        #     plt.xlabel("frame")
        #     plt.ylabel("c4")
        #     # plt.plot(np.arange(len(new_theta)) , new_theta, c="r")
        #
        # plt.figure(2)
        # # plt.scatter(np.arange(len(phi)), phi, c='r', marker='x')
        # plt.plot(phi , "o-")
        # # plt.scatter(count, phi[count], c=color, marker='x')
        # plt.xlabel("frame")
        # plt.ylabel("$tan^{-1}$(c1/c2)")
        # plt.show()