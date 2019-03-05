import numpy as np
import cv2


x,y=np.meshgrid(np.arange(-960,960,1),np.arange(-540,540,1))
f=500
p=8e-3

color = cv2.imread('color.jpg',0)
color=color.astype(np.float32)
Acomlx = 0
Acomlx1 = 0
lmd=632e-6


k = 2 * np.pi / lmd
for u in range(0,99, 5):
    for v in range(0,99,5):
        A=color[v,u]
        d=630
        v1=50-v
        u1=u-50
        Acomlx += A * np.exp(-k * 1j * (np.sqrt((x * 8e-3-u1*0.15) ** 2 + (-y * 8e-3-v1*0.15) ** 2 + d ** 2)-np.sqrt(450**2+(x*8e-3)**2+(y*8e-3)**2)))
#            Acomlx = Acomlx + A
        print v

w=np.reshape(np.random.rand(10000), (100,100))
b=np.reshape(np.random.rand(100*100), (100,100))
color1=w*color+b


for u in range(0,99, 5):
    for v in range(0,99,5):
        A=color1[v,u]
        d=500
        v1=50-v
        u1=u-50
        Acomlx1 += A * np.exp(-k * 1j * (np.sqrt((x * 8e-3-u1*0.15) ** 2 + (-y * 8e-3-v1*0.15) ** 2 + d ** 2)-np.sqrt(450**2+(x*8e-3)**2+(y*8e-3)**2)))
#            Acomlx = Acomlx + A
        print v

E = Acomlx1/Acomlx
z = np.angle(E)/np.pi
z1=(z+1)/2*255
z2=z1.astype("uint8")
print z


color = cv2.imread('depth.jpg',0)
color=color.astype(np.float32)
Acomlx = 0
Acomlx1 = 0
for u in range(0,99, 5):
    for v in range(0,99,5):
        A=color[v,u]
        d=630
        v1=50-v
        u1=u-50
        Acomlx += A * np.exp(-k * 1j * (np.sqrt((x * 8e-3-u1*0.15) ** 2 + (-y * 8e-3-v1*0.15) ** 2 + d ** 2)-np.sqrt(450**2+(x*8e-3)**2+(y*8e-3)**2)))
#            Acomlx = Acomlx + A
        print v


color1=w*color+b

for u in range(0,99, 5):
    for v in range(0,99,5):
        A=color1[v,u]
        d=500
        v1=50-v
        u1=u-50
        Acomlx1 += A * np.exp(-k * 1j * (np.sqrt((x * 8e-3-u1*0.15) ** 2 + (-y * 8e-3-v1*0.15) ** 2 + d ** 2)-np.sqrt(450**2+(x*8e-3)**2+(y*8e-3)**2)))
#            Acomlx = Acomlx + A
        print u,v
E = Acomlx1/Acomlx
z = np.angle(E)/np.pi
z1=(z+1)/2*255
z3=z1.astype("uint8")
print z
cv2.imwrite('wxb.bmp', z2)
cv2.imwrite('wxb1.bmp', z3)

