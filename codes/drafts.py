
##########################################
##########################################
##########################################

def deep(x, A, B):
    a = 0.3
    b = (-A[1] - a*B[0]**2)/(B[0])
    y = a*x**2 + b*x + A[1]
    return y

def steep(x, A, B):
    a = np.sqrt(B[0])*np.sqrt(B[0]*A[1] + 4.)/(2.*np.sqrt(A[1])) - B[0]/2.
    c = A[1] - 1./a
    y = 1./(x+a) + c
    return y



xline = np.linspace(pointA[0], pointB[0], 1000)
y_straight = straight(xline, pointA, pointB)
y_deep = deep(xline, pointA, pointB)
y_steep = steep(xline, pointA, pointB)
y_inverse = inverse(xline, pointA, pointB)

tline = np.linspace(0,2,1000)
x_straight = xstraight(tline, pointA, pointB)
t_straight = tstraight(xline, pointA, pointB)

plt.figure()
plt.plot(tline, x_straight)
plt.plot(t_straight, xline)

plt.clf()
plt.plot(xline, y_straight, label="straight", color='C3')
plt.plot(xline, y_deep, label="deep", color='C4')
plt.plot(xline, y_steep, label="steep", color='C5')
plt.hlines(0, -1, 11, color='k', linestyles='--')
plt.vlines(0, -4, 11, color='k', linestyles='--')
plt.xlim(-1, 11)
plt.ylim(-4, 11)
plt.plot(pointA[0], pointA[1], 's', color='C1')
plt.plot(pointB[0], pointB[1], 's', color='C2')
plt.grid()
plt.xlabel(r"distance $x$ [m]")
plt.ylabel(r"height $y$ [m]")

plt.legend()
