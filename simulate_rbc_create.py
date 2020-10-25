import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import struct
import os
from piradon import iradon

# import/export the optical field
def exportPM(path, reconFlag, angX, angY, nr, nc, phasemap):
    img = np.zeros((int(nr),int(nc)), dtype=np.float32).flat
    phasemap = phasemap.astype(np.float32).flat
    img[phasemap>0]=0.01
    
    with open(path,'wb+') as f:
        f.write(struct.pack('<?ddii', reconFlag, angX, angY, nr, nc))
        for i in range(len(phasemap)):
            f.write(struct.pack('<ff', phasemap[i], img[i]))

c0 = 0.2072
c1 = 2.0026
c2 = -1.1228

n_rbc = 1.395
n_med = 1.334
R = 3.9              #(um)
wavelength = 0.532    #(um)

dx = 5.5             #(um)
mag = 85
grids = round((R*2) / (dx/mag))    # Ngrid= (2 x radius) / actural size per pixel
cmap = plt.cm.gray

plug_no = 2    #1: opticspy; 2: prysm

S = np.zeros((grids, grids))
halfGrids = grids // 2
for y in range(-halfGrids, halfGrids):
    for x in range(-halfGrids, halfGrids):
        px = x + halfGrids
        py = y + halfGrids
        r = np.sqrt(x * x + y * y) * (dx / mag)
        if r < R:
            S[py, px] = np.sqrt(1 - np.power((r / R), 2)) * (
                        c0 + c1 * np.power((r / R), 2) + c2 * np.power((r / R), 4)) * R
        else:
            S[py, px] = 0

phasemap = np.zeros_like(S)
phasemap = S * 2 * np.pi / wavelength * (n_rbc - n_med)
S_3D = np.zeros((grids, grids, grids)).astype(np.float)
S_3D[:, :, :] = 0
halfGrids = grids // 2

for y in range(-halfGrids, halfGrids):
    for x in range(-halfGrids, halfGrids):
        px = x + halfGrids
        py = y + halfGrids
        r = np.sqrt(x * x + y * y)
        if r < halfGrids:
            rz = int(0.5 * halfGrids * np.sqrt(1 - (x * x + y * y) / (halfGrids * halfGrids)) * (
                        c0 + c1 * (x * x + y * y) / (halfGrids * halfGrids) + c2 * np.power((x * x + y * y),2) / np.power(halfGrids,4)))
            S_3D[px, py, halfGrids - rz:halfGrids + rz] = n_rbc

dt_PM = np.dtype({'names': ['status', 'angleX', 'angleY', 'sizeX', 'sizeY', 'cImg'],
               'formats': ['?', '<f8', '<f8', '<i4', '<i4', '(1024,1024,2)<f4']});


import cupy
from cupyx.scipy import ndimage
from scipy.ndimage import rotate

rotation = np.linspace(0,360, num=36, endpoint=False)

nrows = 6
ncols = 6
# fig, axes = plt.subplots(nrows=nrows6, ncols=ncols, figsize=(12, 12), sharex=True, sharey=True)

phasemap_list = []
zc_list = []
nr = 121
nc = 121
tmpPM = np.zeros((nr, nc))
sr = (nr - grids) // 2
er = sr + grids
sc = (nc - grids) // 2
ec = sc + grids

outputDir = r"D:\lab\CODE\rbc_img_analyze\simulate_rbc\test"
count = 0
phimap_stack = np.zeros((nr,nc,len(rotation)),dtype=np.float32)
deg = []

count=0
for theta in tqdm(np.nditer(rotation)):
    newe_S = rotate(S_3D, theta, axes=(1,2), reshape=False)
    
    proj = np.sum(newe_S, axis = 2)*(1/(2*np.pi/wavelength))
    phasemap = proj*2*np.pi/wavelength*(n_rbc-n_med)
    phimap_stack[:,:,count]=phasemap
    deg.append(theta*np.pi/180)
    
    # if not os.path.exists(outputDir):
    #     os.makedirs(outputDir)
    # SavePath = '%s/buffer%03d.phimap' %(outputDir,count)
    # exportPM(SavePath, True, theta*np.pi/180, 0.0, nr, nc, tmpPM)
    count = count+1

np.save(r"D:\lab\CODE\rbc_img_analyze\simulate_rbc\phimap_stack",phimap_stack)
np.save(r"D:\lab\CODE\rbc_img_analyze\simulate_rbc\deg",deg)
