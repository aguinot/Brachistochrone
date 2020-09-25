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

plt.figure()
plt.clf()
plt.plot(0, 0, "ko")
plt.fill_between([-10, 10], 0, -1, color="C5", alpha=0.3)
plt.plot(5, -2, "rs")
plt.xlim(-3, 7)
plt.ylim(-2.5, 0.5)
