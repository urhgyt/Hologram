import numpy as np
import cv2


u,v=np.meshgrid(np.arange(-960,960,1),np.arange(-540,540,1))
x,y=np.meshgrid(np.arange(-960,960,1),np.arange(-540,540,1))
p=8e-3
d=630
lmd = 632e-6
k = 2 * np.pi / lmd
px = lmd * d / (p * 1920)
py = lmd * d / (p * 1080)

slm1 = cv2.imread('SLM1.bmp',0)
slm2 = cv2.imread('SLM2.bmp',0)
slm1 = slm1.astype("float32")
slm1 = (slm1 / 255*2-1)*np.pi

E = np.exp(slm1*1j)


E0 = np.exp(-np.pi * 1j /(lmd*d)* ((x * px) ** 2 + (y * py) ** 2))
E1 = E * np.exp(-k/d * 1j/2 * ((u * p) ** 2 + (v * p) ** 2))

Ef = np.fft.ifftshift(np.fft.ifft2(E1))*E0
simul1 = np.abs(Ef)
smin, smax = simul1.min(), simul1.max()
simul1 = (simul1-smin)/(smax-smin)*255             #guiyihua
simul1=simul1.astype("uint8")
cv2.imwrite('SIMUL11.bmp', simul1)

