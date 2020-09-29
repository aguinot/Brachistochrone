%matplotlib widget

from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider, Button
import ipywidgets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

A = (0, 0)
B = (5, -2)
D = (0, -1)
v1 = 1.0/2.0
v2 = 1.0

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=[5, 5])
ax[0].plot(0, 0, "ko")
ax[0].fill_between([-10, 10], 0, D[1], color="C5", alpha=0.3)
ax[0].plot(5, -2, "rx")
ax[0].set_xlim(-1, 6)
ax[0].set_ylim(-2.3, 0.5)
ax[1].set_xlabel(r"position $x \,\rm{[m]}$", fontweight='bold')
ax[0].set_ylabel(r"position $y \,\rm{[m]}$", fontweight='bold')
ax[1].set_ylabel(r"time $t \,\rm{[s]}$", fontweight='bold')
arr_sleigh = mpimg.imread('images/sleigh.png')
arr_igloo = mpimg.imread('images/igloo.png')
ax[0].imshow(arr_sleigh, extent=[-0.2, 0.2, -0.0, 0.4])
ax[0].imshow(arr_igloo, extent=[B[0]-0.2, B[0]+0.2, B[1]-0.0, B[1]+0.4])
ax[0].text(-0.9, -0.5, r"Mud", fontweight='bold')
ax[0].text(-0.9, -1.7, r"Snow", fontweight='bold')
l1, = ax[0].plot([A[0], D[0]], [A[1], D[1]], "--", color="C2", lw=2)
l2, = ax[0].plot([D[0], B[0]], [D[1], B[1]], "--", color="C2", lw=2)
t1 = ax[0].text((A[0]+ A[0])/2.+0.1, -0.5, r"$d_m=1$")
t2 = ax[0].text((A[0]+ B[0])/2.+0.1, -1.5, r"$d_s=%.1f$" % (np.sqrt(1+25)))
paim, = ax[0].plot(0, -1, "r*")

ax[1].set_xlim(-1, 6)
ax[1].set_ylim(6, 12)
ax[1].grid()
xx = 0.0
d1 = np.sqrt(1 + xx**2)
d2 = np.sqrt(1 + (B[0]-xx)**2)
lt, = ax[1].plot(xx, d1/v1 + d2/v2, "o", color="C4")
listp = []


def update(X=0.0):
    xx=X
    x = X
    paim.set_xdata([x])
    l1.set_xdata([A[0], x])
    l2.set_xdata([x, B[0]])
    t1.set_position([(x+A[0])/2.+0.1, -0.5])
    t2.set_position([(x+B[0])/2.+0.1, -1.5])
    d1 = np.sqrt(1 + x**2)
    d2 = np.sqrt(1 + (B[0]-x)**2)
    t1.set_text(r"$d_m=%.1f$ m" %d1)
    t2.set_text(r"$d_s=%.1f$ m" %d2)
    listp.append(ax[1].plot(x, d1/v1 + d2/v2, ".", color="C1"))
    lt.set_xdata(x)
    lt.set_ydata(d1/v1 + d2/v2)
    ax[0].figure.canvas.draw()
    ax[1].figure.canvas.draw()
    # fig.canvas.draw_idle()

def resetplot(z=True):
    for aa in listp:
        aa[0].remove()
    listp[:] = []
    # listp.append(ax[1].plot(xx, d1/v1 + d2/v2, ".", color="C1"))
    ax[0].figure.canvas.draw()
    ax[1].figure.canvas.draw()
    return lt

inter = interact(update, X=FloatSlider(min=-1, max=5.0, step=0.2, continuous_update=True))
button = Button(description='reset')
button.on_click(resetplot)
ipywidgets.VBox([button])
