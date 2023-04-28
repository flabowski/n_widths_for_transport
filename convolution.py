# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:01:33 2022

@author: florianma
"""
from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, orth, qr
from scipy.optimize import curve_fit
from numpy import sin, cos, pi
cmap = plt.cm.plasma
from initial_conditions import Domain, Heaviside, LinearRamp, SmoothRamp, Sigmoid, CkRamp
from basis_functions import SVD, Trigonometric, Greedy, LNA, LPF, Sinc

mult = 1

m, n, r = 10*mult, 10*mult, 10*mult
x = Domain([0, 1], m)
x4 = Domain([0, 4], m*4)
mu = Domain([0, 1], n)
mu4 = Domain([0, 4], n*4)

u_lr = Heaviside()

u_k = u_lr
X = u_k(x(), mu())

svd_basis = SVD(X)
U = svd_basis.U
S = svd_basis.S
VT = svd_basis.VT


U4 = np.zeros((m*4, r))
VT4 = np.zeros((r, n*4))
S4 = S[:r]
T = x.max-x.min
delta_x = x.delta_x
A = 2*delta_x**0.5 * np.sin(np.pi/4)
A = (2*delta_x)**.5

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
for i in range(r):
    omega = 2 * np.pi/(4*T) * (2*i+1)
    U4[:, i] = A * np.sin(omega*x4())
    VT4[i, :] = A * np.cos(omega*mu4())
    if i < 10:
        ax1.plot(x(), U[:, i], "o-")
        ax1.plot(x4(), U4[:, i], ".-")
        ax2.plot(mu(), VT[i, :])
        ax2.plot(mu4(), VT4[i, :])
plt.show()

X4_reconstructed = U4 @ (S4[:, None]*VT4)
fig, ax = plt.subplots()
ax.imshow(X4_reconstructed, interpolation="nearest", vmin=-1.0, vmax=1.0)
plt.show()

X4_repaired = np.round(X4_reconstructed*2, decimals=0)/2
fig, ax = plt.subplots()
ax.imshow(X4_repaired, interpolation="nearest", vmin=-1.0, vmax=1.0)
plt.show()

fig, ax_ss = plt.subplots()
fig, ax_S = plt.subplots()
X4_smoothened = X4_repaired.copy()
kernel = np.ones((3, 3), dtype=np.float64)
kernel /= kernel.size
for j in range(6):
    print(j)
    fig, ax = plt.subplots()
    ax.imshow(X4_smoothened, interpolation="nearest", vmin=-1.0, vmax=1.0)
    plt.show()
    for i in [0*mult, 5*mult, 9*mult]:
        ax_ss.plot(x(), X[:, i], "o-")
        ax_ss.plot(x4(), X4_smoothened[:, i], ".-")
        # plt.plot(x4(), X44[:, i], ".-")

    X4_smoothened = convolve2d(X4_smoothened, kernel,
                               boundary='symm', mode='same')

    svd_basis = SVD(X4_smoothened)
    U44 = svd_basis.U
    S44 = svd_basis.S
    VT44 = svd_basis.VT

    ax_S.plot(S44)

    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # for i in range(10):
    #     ax1.plot(x(), U[:, i]/2, "k-.")
    #     ax1.plot(x4(), U44[:, i], ".-")
    #     ax2.plot(mu(), VT[i, :]/2, "k-.")
    #     ax2.plot(mu4(), VT44[i, :], ".-")
    # plt.show()
plt.show()


ax_S.set_yscale('log')
ax_S.set_title("S")
# plt.legend(prop={'size': 8})
ax_S.set_xlabel(r'$N$')
ax_S.set_ylabel('S')
plt.show()