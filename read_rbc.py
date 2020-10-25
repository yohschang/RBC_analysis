# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 21:27:56 2020

@author: YX
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# for i in range(52,499,1):
#     img = np.load(r"D:\data\2020-08-13\60x\20\phi"+"\\"+str(i)+".npy")
#     plt.imshow(img.real,cmap="jet",vmin = 0,vmax = 3)
#     plt.title(str(i))
#     plt.show()

db = pd.read_pickle(r"D:\data\2020-09-17\60x_2\phi\rbc1.pickle")
for i in range(db["rbc_array"].size):
    print(i)
    img = db["rbc_array"].iloc[i]
    ms = db["mass_center"].iloc[i]
    plt.imshow(img , cmap="jet" , vmin = 0 , vmax = 3)
    plt.title(ms)
    plt.show()