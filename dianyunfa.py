import numpy as np
import cv2


#print color.dtype


x,y=np.meshgrid(np.arange(-540,540,1),np.arange(-540,540,1))

print x.shape
print x.dtype
# f=280
#
p=8e-3
# ps=0.15
# k=2*np.pi/lmd
# Acomlx = 0
# f1=450
lmd = 632e-6
d=p**2*1080/lmd
k = 2 * np.pi / lmd
u,v=np.meshgrid(np.arange(-540,540,1),np.arange(-540,540,1))

# def dianyun(x,y,color,d):
#     Acomlx = 0
#     lmd=632e-6
#     k = 2 * np.pi / lmd
#     for u in range(0,99, 1):
#         for v in range(0,99,1):
#             A=color[v,u]
#             v1=50-v
#             u1=u-50
#             Acomlx += A * np.exp(-k * 1j * (np.sqrt((x * 8e-3-u1*0.15) ** 2 + (-y * 8e-3-v1*0.15) ** 2 + d ** 2)-np.sqrt(450**2+(x*8e-3)**2+(y*8e-3)**2)))
# #            Acomlx = Acomlx + A
#             print v
#     return Acomlx
#     pass
ran=np.exp(2*np.pi*1j*np.reshape(np.random.rand(1080*1080), (1080, 1080)))
#ran = np.exp(np.pi*1j/4)
color2= cv2.imread('b.bmp', 0)
color2= cv2.resize(color2,(1080,1080))
color2 = (color2/255*254+1)/255
color2 = color2 * ran
px = lmd * d / (p * 1080)
py = lmd * d / (p * 1080)
#d= px/lmd*(p*1920)
E0 = color2 * np.exp(np.pi * 1j / (lmd * d) * ((x * px) ** 2 + (y * py) ** 2))
E1 = np.exp(k / d * 1j / 2 * ((u * p) ** 2 + (v * p) ** 2))
#Acomlx=Acomlx/np.exp(-1j*np.sqrt(f1**2+(x*p)**2+(y*p)**2))

Acomlx=np.fft.fftshift(np.fft.fft2(E0))*E1
z=-np.angle(Acomlx)/np.pi
z1=(z+1)/2*255
z2=z1.astype("uint8")
cv2.imwrite('SLM1.bmp', z2)

z1=z2.astype('float32')
z2=(z1/255*2-1)*np.pi

Acomlx1 = np.exp(-z2*1j)

E0 = np.exp(-np.pi * 1j /(lmd*d)* ((x * px) ** 2 + (y * py) ** 2))
E1 = Acomlx1 * np.exp(-k/d * 1j/2 * ((u * p) ** 2 + (v * p) ** 2))

E1 = np.fft.ifft2(E1)
#E1 = np.fft.ifftshift(E1)

U1 = E1*E0
simul1= np.abs(U1)
smin, smax = simul1.min(), simul1.max()
simul1 = (simul1-smin)/(smax-smin)*255             #guiyihua
simul1=simul1.astype("uint8")
cv2.imwrite('SIMUL1.bmp', simul1)

color= cv2.imread('a.bmp', 0)
color= cv2.resize(color,(1080,1080))
color = (color/255*254+1)/255
color = color * ran


E0 = color * np.exp(np.pi * 1j /(lmd*d)* ((x * px) ** 2 + (y * py) ** 2))
E1 = np.exp(k/d * 1j/2 * ((u * p) ** 2 + (v * p) ** 2))

E0= np.fft.fftshift(np.fft.fft2(E0))
#    Ef2=reikna_fft(E0,1080,1920)

U2 = E0*E1
print U1.shape
# r=np.sqrt((u * p) ** 2 + (v * p) ** 2+d**2)
# U1 = np.exp(1j*k*r)

F = U2/U1

z = np.real(F)
z1=(z/np.pi+1)/2*255

z2=z1.astype("uint8")

cv2.imwrite('SLM2.bmp', z2)

z1=z2.astype('float32')
z2=(z1/255*2-1)*np.pi
print z2
# F = np.exp(-z*1j)
F = z
E0 = np.exp(-np.pi * 1j /(lmd*d)* ((x * px) ** 2 + (y * py) ** 2))
E1 = U1 * F * np.exp(-k/d * 1j/2 * ((u * p) ** 2 + (v * p) ** 2))

simul2 = np.fft.ifft2(E1)*E0
simul2= np.abs(simul2)
smin1, smax1 = simul2.min(), simul2.max()
simul2 = (simul2-smin1)/(smax1-smin1)*255              #guiyihua
simul2=simul2.astype("uint8")
cv2.imwrite('SIMUL2.bmp', simul2)