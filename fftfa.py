import numpy as np
import cv2

from timeit import default_timer as timer
#import cudafft2d as cuda



#color= cv2.imread('images/a.bmp',0)
#color=(255-color)/255
#color=color.astype("complex128")
ran=np.exp(2*np.pi*1j*np.reshape(np.random.rand(1920*1080), (1080, 1920)))
#ran=1

x,y=np.meshgrid(np.arange(-960,960,1),np.arange(-540,540,1))
u,v=np.meshgrid(np.arange(-960,960,1),np.arange(-540,540,1))
f=[400,450,500]
lmd=632e-6
p=8e-3

k=2*np.pi/lmd
colors=['a.bmp']

start = timer()
i=0
color = cv2.imread(colors[i], 0)

# w=np.reshape(np.random.rand(1920*1080), (1080,1920))
# b=np.reshape(np.random.rand(1920*1080), (1080,1920))
# color1=w*color+b

color = (255 - color) / 255
color = color * ran


px = lmd * f[i] / (p * 1920)
py = lmd * f[i] / (p * 1080)
E0 = np.exp(-np.pi * 1j /(lmd*f[i])* ((x * px) ** 2 + (y * py) ** 2))
E1 = np.exp(-k/f[i] * 1j/2 * ((u * p) ** 2 + (v * p) ** 2))
print f[i]

Ef= np.fft.fftshift(np.fft.fft2(E0))
#    Ef2=reikna_fft(E0,1080,1920)

E = Ef*E1
print E

r=np.sqrt((u * p) ** 2 + (v * p) ** 2+450**2)
E =E/(color*np.exp(1j*k*r))
z = np.angle(E)/np.pi
z1=(z+1)/2*255
z1=z1.astype("uint8")
timeit=timer()-start
print("fft took %f seconds " % timeit)
print z1
cv2.imwrite('fftpingxingabc.bmp', z1)
cv2.imshow("tupian",z1)
cv2.waitKey()
