# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:34:09 2022

@author: florianma
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, orth, qr
from scipy.optimize import curve_fit
from numpy import sin, cos, pi

from initial_conditions import Domain, Heaviside, LinearRamp, SmoothRamp, Sigmoid, CkRamp
from basis_functions import SVD, Trigonometric, Greedy, LNA, LPF

cmap = plt.cm.plasma
m, n, r = 50, 20, 3
x = Domain([0, 1], m)
mu = Domain([-.0, 1.0], n)

u_hs = Sigmoid(50)
# u_hs.plot(x(), label=u_hs.name)
# plt.legend(prop={'size': 8})
X = u_hs(x(), mu())

svd_basis = SVD(X)
u_hs.S = svd_basis.S
trig_basis = Trigonometric(x)

U = svd_basis.U
U_trig = trig_basis.U

delta_x = x()[1]-x()[0]
A_ = (2*delta_x)**.5
c_ = 0
T = 1

# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #
a, b, N = m, n, r
U = svd_basis.U[:, :r]
sigma_N_2 = 1/b * np.sum(1/a * np.sum((X - (U@(U.T@X)))**2, axis=0))
# sigma_N_2_ = 1/b * np.sum(1/a * np.sum((X**2 - 2*X*(U@(U.T@X)) + (U@(U.T@X))**2), axis=0))

###############################################################################
approx = X.copy()*0
res = 0.0
C = (U.T@X)
for j in range(b):
    for i in range(a):
        for r in range(N):
            approx[i, j] += C[r, j]*U[i, r]
        res += 1/(a*b)*(X[i, j]-approx[i, j])**2
print(sigma_N_2-res)
###############################################################################
approx = X.copy()*0
res = 0.0
C = (U.T@X)
for j in range(b):  # for each snapshot
    for i in range(a):  # for each x
        for r in range(N):  # compute approx
            for k in range(a):  # compute coeff
                c_rj = U[k, r]*X[k, j]
                approx[i, j] += U[k, r]*X[k, j]*U[i, r]
        t1 = X[i, j]**2
        t2 = -2*X[i, j]*approx[i, j]
        t3 = approx[i, j]**2
        res += 1/(a*b)*(t1+t2+t3)
print(sigma_N_2-res)
###############################################################################
t1 = t2 = t3 = 0.0
approx = X.copy()*0
res = 0.0
C = (U.T@X)
for j in range(b):  # for each snapshot
    for i in range(a):  # for each x
        for r in range(N):  # compute approx
            for k in range(a):  # compute coeff
                c_rj = U[k, r]*X[k, j]
                approx[i, j] += U[k, r]*X[k, j]*U[i, r]
        t1 += X[i, j]**2
        t2 += -2*X[i, j]*approx[i, j]
        t3 += approx[i, j]**2
res = 1/(a*b)*(t1+t2+t3)
print(sigma_N_2-res)
print(t1, t2, t3)
###############################################################################
t1 = 0.0
for j in range(b):  # for each snapshot
    for i in range(a):  # for each x
        t1 += X[i, j]**2
t2 = 0
for r in range(N):  # compute approx
    for j in range(b):  # for each snapshot
        for i in range(a):  # for each x
            c_rj = 0.0
            for k in range(a):  # compute coeff
                c_rj += U[k, r]*X[k, j]
            # approx[i, j] += c_rj*U[i, r]
            t2 += -2*X[i, j]*U[i, r]*c_rj

# U@U.T@X * U@U.T@X = U@C * U@C:
t3 = 0
approx2 = X.copy()*0
for j in range(b):  # for each snapshot
    for i in range(a):  # for each x
        for r in range(N):  # compute approx
            for s in range(N):  # compute approx
                approx2[i, j] += U[i, r]*U[i, s] * C[r, j]*C[r, j]
t3 = np.sum(approx2)
print(t1, t2, t3)
###############################################################################
t2 = 0
for r in range(N):  # compute approx
    for j in range(b):  # for each snapshot
        rs = 0
        for i in range(a):  # for each x
            rs += (X[i, j]*U[i, r])
        t2 += -2*rs**2
t3 = 0
for r in range(N):  # compute approx
    for j in range(b):  # for each snapshot
        t3 += C[r, j]**2
print(t1, t2, t3)

np.sum(X*X)
np.sum(X*approx)
np.sum(approx**2)
np.sum(C**2)
asd

dp = 0
for i in range(a):  # for each x
    for r in range(N):  # compute approx
        for s in range(N):  # compute approx
            dp += U[i, r]*U[i, s]
print(dp)
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - #


def func(x, A, omega, c):
    return A*np.sin(omega*x-c)


fig, ax = plt.subplots()
for i in range(15):
    omega_ = 2 * np.pi/(4*T) * (2*i+1)
    yy = U[:, i]
    ll = (0 <= x()) & (x() <= 1)
    popt, pcov = curve_fit(func, x()[ll], yy[ll], [A_, omega_, c_])
    A, omega, c = popt
    print(omega, c)
    y_fit = func(x(), A, omega, c)

    lbl = "mode {:.0f}".format(i)
    plt.plot(x(), U[:, i], "o", ms=1, color=cmap(i/14), label=lbl)
    ax.plot(x(), y_fit, "r--", lw=.5)  # label="mode {:.0f}".format(i))
    # ax.plot(x(), U_trig[:, i], "k--", lw=.5)
ax.plot([-2, -1], [0, 0], "k--", lw=.5, label="theoretical results")
plt.legend(prop={'size': 6})
plt.grid(which="both")
plt.xlim([.45, .55])
# plt.ylim([-A, 0.035])
plt.xlabel("x")
plt.show()