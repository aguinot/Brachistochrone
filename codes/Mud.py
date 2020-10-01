'''
Mud problem
'''

import numpy as np
import scipy
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import matplotlib.image as mpimg
from matplotlib import offsetbox
from scipy import optimize
plt.ion()
plt.show()

A = (0, 0)
B = (5, -2)
D = (0, -1)

#### static plot :
fig, ax = plt.subplots(figsize=[10, 5])
plt.clf()
ax = plt.gca()
plt.plot(0, 0, "ko")
plt.fill_between([-10, 10], 0, -1, color="C5", alpha=0.3)
plt.plot(5, -2, "rx")
plt.xlim(-1, 6)
plt.ylim(-2.3, 0.5)
plt.xlabel(r"x", fontweight='bold')
plt.ylabel(r"y", fontweight='bold')


arr_sleigh = mpimg.imread('images/sleigh.png')
arr_igloo = mpimg.imread('images/igloo.png')
ax.imshow(arr_sleigh, extent=[-0.2, 0.2, -0.0, 0.4])
ax.imshow(arr_igloo, extent=[B[0]-0.2, B[0]+0.2, B[1]-0.0, B[1]+0.4])
# Smashicons
# Pixelmeetup
plt.text(-0.9, -0.5, r"Mud", fontweight='bold')
plt.text(-0.9, -1.7, r"Snow", fontweight='bold')
l1, = ax.plot([A[0], D[0]], [A[1], D[1]], "--", color="C2", lw=2)
l2, = ax.plot([D[0], B[0]], [D[1], B[1]], "--", color="C2", lw=2)
plt.savefig("images/mud1.jpg")


#### interactive plot :



fig, ax = plt.subplots(figsize=[10, 5])
plt.clf()
ax = plt.gca()
plt.plot(0, 0, "ko")
plt.fill_between([-10, 10], 0, -1, color="C5", alpha=0.3)
plt.plot(5, -2, "rx")
plt.xlim(-1, 6)
plt.ylim(-2.3, 0.5)
plt.xlabel(r"x", fontweight='bold')
plt.ylabel(r"y", fontweight='bold')
arr_sleigh = mpimg.imread('images/sleigh.png')
arr_igloo = mpimg.imread('images/igloo.png')
ax.imshow(arr_sleigh, extent=[-0.2, 0.2, -0.0, 0.4])
ax.imshow(arr_igloo, extent=[B[0]-0.2, B[0]+0.2, B[1]-0.0, B[1]+0.4])
plt.text(-0.9, -0.5, r"Mud", fontweight='bold')
plt.text(-0.9, -1.7, r"Snow", fontweight='bold')
plt.plot([A[0], D[0]], [A[1], D[1]], "--", color="C3", lw=2)
plt.plot([D[0], B[0]], [D[1], B[1]], "--", color="C2", lw=2)


# plt.savefig("images/mud1.jpg")


#### general  plot :

fig, ax = plt.subplots(figsize=[10, 5])
plt.clf()
ax = plt.gca()
plt.plot(0, 0, "ko")
plt.fill_between([-10, 10], 0, -1, color="C5", alpha=0.3)
plt.plot(5, -2, "rx")
plt.xlim(-1, 6)
plt.ylim(-2.3, 0.5)
plt.xticks([])
plt.yticks([])
#plt.xlabel(r"x", fontweight='bold')
#plt.ylabel(r"y", fontweight='bold')

x = 1.0
arr_sleigh = mpimg.imread('images/sleigh.png')
arr_igloo = mpimg.imread('images/igloo.png')
#ax.imshow(arr_sleigh, extent=[-0.2, 0.2, -0.0, 0.4])
#ax.imshow(arr_igloo, extent=[B[0]-0.2, B[0]+0.2, B[1]-0.0, B[1]+0.4])
#plt.text(-0.9, -0.5, r"Mud", fontweight='bold')
#plt.text(-0.9, -1.7, r"Snow", fontweight='bold')
l1, = ax.plot([A[0], x], [A[1], D[1]], "--", color="C2", lw=2)
l2, = ax.plot([x, B[0]], [D[1], B[1]], "--", color="C2", lw=2)
plt.vlines(x, , ls=":", color="k", lw=1)
# plt.savefig("images/mudgeneral.jpg")



#### general time plot  :

fig, ax = plt.subplots(figsize=[10, 4])
plt.clf()
ax = plt.gca()
xline = np.linspace(-1, 5, 1000)
v1 = 1.0/2.0
v2 = 1.0
a = 1
b = 2
m = 5
time = np.sqrt(a*a + xline*xline)/v1 +  np.sqrt((m-xline)**2 + (b-a)**2)/v2
plt.plot(xline, time)
ax.set_xlabel(r"position $x \,\rm{[m]}$", fontweight='bold')
ax.set_ylabel(r"time $t \,\rm{[s]}$", fontweight='bold')
ax.text(3, 9, r"$t(x)$")
plt.savefig("images/generaltime.jpg")


xline = np.linspace(0.5, 0.6, 10000000)
time = np.sqrt(a*a + xline*xline)/v1 +  np.sqrt((m-xline)**2 + (b-a)**2)/v2
xline[np.argmin(time)]
