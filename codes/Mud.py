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
fig, ax = plt.subplots(figsize=[10, 5])
plt.clf()
ax = plt.gca()
# plt.plot(0, 0, "ko")
plt.fill_between([-10, 10], 0, -1, color="C5", alpha=0.3)
# plt.plot(5, -2, "rs")
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
plt.savefig("images/mud1.jpg")
