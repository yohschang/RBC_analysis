# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 18:16:15 2020

@author: YX
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as cm
# import cupy
# import interferometer_zenike as __interferometer__

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
		"""
		------------------------------------------------
		listcoefficient():
		List the coefficient in Coefficient
		-----------------------------------------------
		"""
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
		"""
		------------------------------------------------
		zernikelist():
		List all Zernike Polynomials
		------------------------------------------------
		"""
		m = 1
		for i in self.__zernikelist__:
			print("Z"+str(m)+":"+i)
			m = m + 1

	def zernikemap(self, label = True):
		"""
		------------------------------------------------
		zernikemap(self, label_1 = True):
		Return a 2D Zernike Polynomials map figure
		label: default show label
		------------------------------------------------
		"""
		theta = np.linspace(0, 2*np.pi, 400)
		rho = np.linspace(0, 1, 400)
		[u,r] = np.meshgrid(theta,rho)
		X = r*np.cos(u)
		Y = r*np.sin(u)
		Z = __zernikepolar__(self.__coefficients__,r,u)
		fig = plt.figure(figsize=(12, 8), dpi=80)
		ax = fig.gca()
		im = plt.pcolormesh(X, Y, Z, cmap=cm.RdYlGn)

		if label == True:
			plt.title('Zernike Polynomials Surface Heat Map',fontsize=18)
			ax.set_xlabel(self.listcoefficient()[1],fontsize=18)
		plt.colorbar()
		ax.set_aspect('equal', 'datalim')
		plt.show()
		return Z


def __zernikepolar__(coefficient,r,u):
	"""
	------------------------------------------------
	__zernikepolar__(coefficient,r,u):
	Return combined aberration
	Zernike Polynomials Caculation in polar coordinates
	coefficient: Zernike Polynomials Coefficient from input
	r: rho in polar coordinates  (equal to 2r in paper)
	u: theta in polar coordinates
    
    full math term =RMS*Polar form
	------------------------------------------------
	"""
	Z = [0]+coefficient
	Z1  =  Z[1]  * 1*(np.cos(u)**2+np.sin(u)**2)
	Z2  =  Z[2]  * 2*r*np.cos(u)
	Z3  =  Z[3]  * 2*r*np.sin(u)
	Z4  =  Z[4]  * np.sqrt(3)*(2*r**2-1)
	Z5  =  Z[5]  * np.sqrt(6)*r**2*np.sin(2*u)
	Z6  =  Z[6]  * np.sqrt(6)*r**2*np.cos(2*u)
	Z7  =  Z[7]  * np.sqrt(8)*(3*r**2-2)*r*np.sin(u)
	Z8  =  Z[8]  * np.sqrt(8)*(3*r**2-2)*r*np.cos(u)
	Z9  =  Z[9]  * np.sqrt(8)*r**3*np.sin(3*u)
	Z10 =  Z[10] * np.sqrt(8)*r**3*np.cos(3*u)
	Z11 =  Z[11] * np.sqrt(5)*(1-6*r**2+6*r**4)
	Z12 =  Z[12] * np.sqrt(10)*(4*r**2-3)*r**2*np.cos(2*u)
	Z13 =  Z[13] * np.sqrt(10)*(4*r**2-3)*r**2*np.sin(2*u)
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
	for i in range(n):
		C = [0]*i+[1]+[0]*(37-i-1)   # coeff of i'th term become 1 
		ZF = __zernikepolar__(C,r,u)
		for i in range(l):
			for j in range(l):
				if x2[i]**2+y2[j]**2>1:   #within unit circle 
					ZF[i][j]=0
					Z[i][j] = 0
		a = sum(sum(Z*ZF))*2*2/l/l/np.pi   # why this can be seen as coef??
		fitlist.append(np.around(a,3))

	l1 = len(fitlist)
	fitlist = fitlist+[0]*(37-l1)
	Z_new = __zernikepolar__(fitlist,r,u)
	for i in range(l):
		for j in range(l):
			if x2[i]**2+y2[j]**2>1:
				Z_new[i][j]=0

	C = Coefficient(fitlist)  #output zernike Coefficient class

	return fitlist , Z_new
