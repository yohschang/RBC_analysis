# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:12:51 2020

@author: YX
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def func(x,a,b,c,d):
    return a*np.cos(b*x + c)**2+d

y_datain = np.loadtxt(r"D:\LOAD.txt")
y_data = y_datain[0:5]

x_data = np.arange(len(y_data))

bound = ([0.8,0,0,0],[1,1,1,0.2])
# popt , pcov = curve_fit(func , x_data , y_data , bounds=bound)
popt , pcov = curve_fit(func , x_data , y_data , bounds=bound)


plt.plot(x_data,y_data ,"b-")
x_new = np.linspace(0,len(y_data),100)
plt.plot(x_new,func(x_new , *popt), "r-")
plt.show()

plt.figure(1)
plt.plot(x_data , x_data*popt[1]+popt[2], '-o')

