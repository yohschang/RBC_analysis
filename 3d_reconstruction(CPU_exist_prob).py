import numpy as np
from scipy.signal import medfilt
import h5py
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import sys
import math
import time
import cv2
from skimage.transform import resize
from scipy import ndimage
from tqdm import tqdm
import scipy.ndimage

import pandas as pd


dt_PM = np.dtype({'names': ['status', 'angleX', 'angleY', 'sizeX', 'sizeY','cImg'],'formats': ['?', '<f8', '<f8', '<i4', '<i4','(1024,1024,2)<f4']});

def opfile(i):
    # if ss == "pha" :
    #     nam = "Phase"
    # elif ss== "amp":
    #     nam = "Amp"
    i = i+1
    if i < 10:
        filePath = r"D:\data\2020-06-28(fringe_dens&3drbc)\Bead\6\Data"+"\\buffer"+"00"+str(i)+".phimap"
    elif i < 100 and i > 9:
        filePath = r"D:\data\2020-06-28(fringe_dens&3drbc)\Bead\6\Data"+"\\buffer"+str(0)+str(i)+".phimap"
    elif i > 99:
        filePath = r"D:\data\2020-06-28(fringe_dens&3drbc)\Bead\6\Data"+"\\buffer" + str(i) + ".phimap"
    data = np.fromfile(filePath, dtype=dt_PM)
    data = data[0]
    reconFlag = data['status']
    angX = data['angleX']
    angY = data['angleY']
    phiImg = data['cImg'][:, :, 0]
    ampImg = data['cImg'][:,:, 1]

    return reconFlag, angX, angY, phiImg, ampImg



def trd_recon():
    # folder path of optical fields
    # opt_field = './Data/'
    # phimap_stack = np.load(r"D:\data\2020-09-17\60x_3\cell1\recon_phimap.npy",allow_pickle=True)
    # theta = np.load(r"D:\data\2020-09-17\60x_3\cell1\recon_deg.npy",allow_pickle=True)
    phimap_stack = np.load(r"D:\lab\CODE\rbc_img_analyze\simulate_rbc\ideal_stack.npy",allow_pickle=True)
    theta = np.load(r"D:\lab\CODE\rbc_img_analyze\simulate_rbc\theta.npy",allow_pickle=True)
    # frame,nx,ny = phimap_stack.shape
    nx,ny,frame = phimap_stack.shape
    # nx = 4*nx
    # ny = 4*ny
    # nx,ny ,frame= phimap_stack.shape

    n_med = 1.334
    wavelength = 532
    CameraPixelSize = 5.5
    mag = 85
    ffsize = 512
    n_med2 = n_med * n_med
    wavelength = wavelength * 1e-9
    f0 = 1 / wavelength
    fm0 = f0 * n_med
    fm02 = fm0 * fm0
    # ffsize = 256
    FOV = 2048
    dx = (CameraPixelSize * 1e-6) / mag * 1024/ ffsize
    # Wave vector
    k = 2 * np.pi * n_med / wavelength
    k2 = k * k
    # gridfrequency resolution
    df = 1 / (ffsize * dx)
    # rowSize = colSize =ffsize
    # ffsize=512
    f_3D = np.zeros((ffsize, ffsize, ffsize), dtype=np.complex64)  # scattering␣

    F_3D = np.zeros((ffsize, ffsize, ffsize), dtype=np.complex64)  # scattering␣

    F_3Dx = np.zeros((ffsize, ffsize, ffsize), dtype=np.complex64)
    C_3D = np.zeros((ffsize, ffsize, ffsize), dtype=np.int)
    u_sp = np.zeros((ffsize, ffsize), dtype=np.complex64)
    U_rytov = np.zeros((ffsize, ffsize), dtype=np.complex64)

    # for filePath in tqdm(glob.glob(opt_field + '/*')[:500]):
    #     reconFlag, angX, angY, phiImg, ampImg = importOpticalField(filePath)
    # if (reconFlag == True):



    for i in tqdm(range(360)):
        # first = time.time()
        # print("frame: "+str(i))
        fz_err = None
        # reconFlag, angX, angY, phiImg, ampImg = opfile(i)

        angX = theta[i]
        angY = 0
        # print(angX,angY)
        # plt.imshow(phiImg)
        # plt.show()

        # print(angX)
        # if np.std(phiImg) < 0.43:
        # if reconFlag == True:
        # phiImg = np.reshape(phiImg,(1024,1024))
        # ampImg = np.reshape(ampImg,(1024,1024))
        # print(phimap_stack[i,:,:].real.shape , phimap_stack[i,:,:].imag.shape)

        # phiImg = phimap_stack[i,:,:].real
        # ampImg = phimap_stack[i,:,:].imag

        phiImg = phimap_stack[:,:,i]
        ampImg = phiImg*0

        phiImg[np.isnan(phiImg)] = 0
        phiImg[np.isinf(phiImg)] = 0
        ampImg[np.isnan(ampImg)] = 1
        ampImg[np.isinf(ampImg)] = 1
        ampImg[ampImg == 0] = 0.01
        ampImg = np.absolute(ampImg)
        phiImg = ndimage.median_filter(resize(phiImg, (ffsize,ffsize)), size=3)
        ampImg = ndimage.median_filter(resize(ampImg, (ffsize,ffsize)), size=3)

        # plt.figure()
        # plt.imshow(phiImg, cmap="gray")
        # plt.show()

        # phiImg = scipy.ndimage.zoom(phiImg, (nx/ffsize,ny/ffsize))
        # ampImg = scipy.ndimage.zoom(ampImg, (nx/ffsize,ny/ffsize))
        u_sp = np.fft.fftshift(np.log(ampImg) + 1j * phiImg)
        U_rytov = np.fft.fft2(u_sp)

        fx0 = fm0 * np.sin(0)    #正像投影時因為k不變因此不用做出fx0與fy0的位移
        fy0 = fm0 * np.sin(0)
        fz0 = np.sqrt(fm02 - fx0 * fx0 - fy0 * fy0)
        # print(fx0, fy0)
        # sec = time.time()
        # print(str(i))
        for row in range(-ffsize // 2, ffsize // 2):
            for col in range(-ffsize // 2, ffsize // 2):
        # for row in range(-27,-8):
            # for col in range(-27,-26):
                Fx = row * df
                Fy = col * df
                fx = Fx + fx0
                fy = Fy + fy0

                tmp_fz = fm02-(fx*fx+fy*fy)
                if tmp_fz < 0:
                    fz_err = True
                    fz = 0
                else:
                    fz_err = False
                    fz = np.sqrt(tmp_fz)
                    Fz = fz - fz0
                    if fm02*2 < fx*fx+fy*fy:
                        fz_err = True
                        fz = 0

                # Fz = fz - fz0
                Nx = round(Fx/df)
                Ny = round(Fy/df)
                
                # if (fm02 - fx0 * fx0 - fy0 * fy0 >= 0 and fm02 - fx * fx - fy * fy >= 0):
                #     flag = True
                # else:
                #     flag = False
                # angX = np.deg2rad(angX)
                if (not fz_err and Nx >= -ffsize / 2 and Nx < ffsize / 2 and Ny >= -ffsize / 2 and Ny < ffsize / 2):
                    Nz = round(Fz/df)
                    Nx2 = Nx * np.cos(angX) - Nz * np.sin(angX)
                    Nz2 = Nx * np.sin(angX) + Nz * np.cos(angX)

                    #
                    if (Nx2 >= -ffsize/2 and Nx2<ffsize / 2 and Nz2 >= -ffsize / 2 and Nz2<ffsize / 2 ):
                        Nx = int(np.mod(Nx, ffsize))
                        Ny = int(np.mod(Ny, ffsize))
                        Nx2  = int(np.mod(Nx2, ffsize))
                        Nz2  = int(np.mod(Nz2, ffsize))
                        # Nz = int(np.mod(int(np.round(Fz / df)), ffsize))
                        # print(Nx ,Ny, Nz)
                        F_3D[Nx2,Nz2 ,Ny] += (1j * 2 * np.pi * fz) * U_rytov[Nx,Ny]
                        # F_3D[Nx2 ,Ny, Nz2] += (1j *  (1/np.pi) * fz) * U_rytov[Nx , Ny]
                        C_3D[Nx2,Nz2 ,Ny] += 1
                        # print("Nx2:"+str(Nx2)+ ", Ny:"+str(Ny)+", Nz2:"+str(Nz2)+", Nx:"+str(Nx)+", usp_r:"+str(U_rytov[Ny,Nx].real)+", usp_i:"+str(U_rytov[Ny,Ny].imag))

                # fz = np.sqrt(fm02-fx*fx-fy*fy)
                # Fz = fz - fz0
                # Nx = row
                # Ny = col
                # if (fm02 - fx0 * fx0 - fy0 * fy0 >= 0 and fm02 - fx * fx - fy * fy >= 0):
                #     flag = True
                # else:
                #     flag = False
                # if (flag == True and Nx >= -ffsize / 2 and Nx < ffsize / 2 and Ny >= -ffsize / 2 and Ny < ffsize / 2):
                #     Nx = int(np.mod(Nx, ffsize))
                #     Ny = int(np.mod(Ny, ffsize))
                #     Nz = int(np.mod(int(np.round(Fz / df)), ffsize))
                #     # F_3D[Nx, Ny, Nz] += (1j * 4 * np.pi * fz) * U_rytov[Nx, Ny]
                #     F_3D[Nx,Ny,Nz] += ((1j*fz)/np.pi) * U_rytov[Nx, Ny]
                #     C_3D[Nx, Ny, Nz] += 1

    # print(C_3D[0:50,0 ,0])
    F_3Dx[C_3D > 0] = F_3D[C_3D > 0] / C_3D[C_3D > 0]
    F_3D2 = F_3D
    print(np.sum(np.isnan(F_3Dx)))

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex=True,sharey = True)
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    C_3D = np.fft.fftshift(C_3D/C_3D)
    # C_3D = C_3D/C_3D
    # xy = np.log10(np.abs(F_3D[:, :, ffsize // 2]))
    # xz = np.log10(np.abs(F_3D[:, ffsize // 2, :]))
    # yz = np.log10(np.abs(F_3D[ffsize // 2, :, :]))
    xy = C_3D[:, :, ffsize // 2].T
    xz = C_3D[:, ffsize // 2, :].T
    yz = C_3D[ffsize // 2, :, :]
    ax1.set_title("xy")
    ax2.set_title("xz")
    ax3.set_title("yz")
    ax1.imshow(xy)
    ax2.imshow(xz)
    ax3.imshow(yz)
    plt.show()


    f_3D = np.fft.ifftn(F_3Dx / dx)
    F_3D_2 = F_3Dx / dx
    n_3D = np.sqrt(-f_3D / k2 + 1) * n_med

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex=True, sharey=True)
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    C_3D = np.fft.fftshift(C_3D / C_3D)
    # C_3D = C_3D/C_3D
    # xy = np.log10(np.abs(np.fft.fftshift(n_3D)[:, :, ffsize // 2]))
    # xz = np.log10(np.abs(np.fft.fftshift(n_3D)[:, ffsize // 2, :]))
    # yz = np.log10(np.abs(np.fft.fftshift(n_3D)[ffsize // 2, :, :]))
    xy = np.real(f_3D)[:, :, ffsize // 2]
    xz = np.real(f_3D)[:, ffsize // 2, :]
    yz = np.real(f_3D)[ffsize // 2, :, :]
    ax1.set_title("xy")
    ax2.set_title("xz")
    ax3.set_title("yz")
    ax1.imshow(xy)
    ax2.imshow(xz)
    ax3.imshow(yz)
    plt.show()

    # Positive constraint
    n_3D[np.real(n_3D) < n_med] = n_med + 1j * np.imag(n_3D[np.real(n_3D) < n_med])
    n_3D[np.imag(n_3D) < 0] = np.real(n_3D[np.imag(n_3D) < 0])

    for iter in tqdm(range(0, 2)):
        # print("iter: "+str(iter))
        # print(iter)
        f_3D = -k2 * (n_3D * n_3D / n_med2 - 1)
        F_3D = np.fft.fftn(f_3D)
        F_3D[F_3D_2 != 0] = F_3D_2[F_3D_2 != 0]
        f_3D = np.fft.ifftn(F_3D)
        n_3D = np.sqrt(-f_3D / k2 + 1) * n_med
        # Positive constraint
        n_3D[np.real(n_3D) < n_med] = n_med + 1j * np.imag(n_3D[np.real(n_3D) < n_med])
        n_3D[np.imag(n_3D) < 0] = np.real(n_3D[np.imag(n_3D) < 0])

    f_3D = -k2 * (n_3D * n_3D / n_med2 - 1)
    F_3D = np.fft.fftn(f_3D)
    F_3D[F_3D_2 != 0] = F_3D_2[F_3D_2 != 0]
    f_3D = np.fft.ifftn(F_3D)
    n_3D = np.sqrt(-f_3D / k2 + 1) * n_med
    n_3D = np.fft.fftshift(n_3D)
    n_3D[np.real(n_3D) < n_med] = n_med + 1j * np.imag(n_3D[np.real(n_3D) < n_med])
    # proj = np.sum(np.real(n_3D),axis = 2)
    # plt.imshow(proj)
    # plt.show()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharex=False,sharey = False)
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    xy = np.real(n_3D[:, :, ffsize//2])
    xz = np.real(n_3D[:, ffsize//2-1, :])
    yz = np.real(n_3D[ffsize//2-1, :, :])
    im1 = ax1.imshow(xy,cmap = "jet")
    ax1.set_title("xy")
    ax2.imshow(xz.T,cmap = "jet")
    ax2.set_title("xz")
    ax3.imshow(yz.T,cmap = "jet")
    ax3.set_title("yz")
    cbar_ax = fig.add_axes([0.93, 0.23, 0.01, 0.55])
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='vertical').set_label(label='RI', size=14)
    plt.show()
    #
    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex=True,sharey = True)
    # ax1 = axes[0]
    # ax2 = axes[1]
    # ax3 = axes[2]
    # xy = np.log10(np.abs(F_3D2[:, :, ffsize // 2]))
    # xz = np.log10(np.abs(F_3D2[:, ffsize // 2, :]))
    # yz = np.log10(np.abs(F_3D2[ffsize // 2, :, :]))
    # ax1.imshow(xy)
    # ax2.imshow(xz)
    # ax3.imshow(yz)
    plt.show()
    return n_3D

# path = "D:\\lab\\3drecon\\load.mat"
# # path = "D:\\lab\\CODE\\pydata.m"
# data = h5py.File(path,"a")
# # n3d = np.array(data["n3d"])
# spPhase = np.array(data["spPhase"])
# Oamp = np.array(data["Oamp"])
# angleMeasureDegX = np.array(data["angleMeasureDegX"])
# angleMeasureDegY = np.array(data["angleMeasureDegY"])
# status = np.array(data["status"])
# phiImgs = np.transpose(spPhase, (2, 1, 0))
# ampImgs= np.transpose(Oamp, (2, 1, 0))


# ang = pd.read_excel(r"D:\1LAB\cTMD\Bead_2017-03-01\angle.xlsx")
# angX = np.array(ang)
# # angX = angX.T
# print(angX[499])
# angY = np.zeros(500)


# phi = np.load(r"D:\lab\CODE\rbc_img_analyze\simulate_rbc\phi.npy")

n3d = trd_recon()
np.save(r"D:\data\2020-09-17\60x_3\cell1\n3d",n3d)





