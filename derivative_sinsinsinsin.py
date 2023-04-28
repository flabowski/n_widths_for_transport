# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 17:32:28 2023

@author: florianma
"""
import numpy as np
import matplotlib.pyplot as plt
# from decimal import Decimal, getcontext
# getcontext().prec = 8
# import mpmath as mp
from mpmath import mp
mp.dps = 80
DTYPE = mp.mpf


pi = mp.pi
plt.close("all")



eps = mp.mpf(0.025)
mu = mp.mpf(0.25)
a = 10_000
x = np.empty(a, dtype=DTYPE)
y = np.empty(a, dtype=DTYPE)
for i in range(a):
    x[i] = i/mp.mpf(a)

k = 4
m = pi**k/mp.mpf(2**k)
x_ = (x-mu)/m*2/eps
for _ in range(k):
    for i in range(a):
        y[i] = mp.sin(pi/2 * x_[i])
    x_ = y
y[x < mu-m*eps/2] = -1
y[x > mu+m*eps/2] = 1
y = y/2+0.5
    
dy_N = y
sigma = np.empty(a, dtype=DTYPE)
for N in np.arange(1, 25):
    print(N-1, (dy_N**2).mean()**.5)
    if N < 15:
        fig, ax = plt.subplots()
        plt.plot(x, dy_N**2, "k.")
        plt.xlim([0.15, 0.35])
        # plt.ylim([-1, 1])
        ax.set_yscale('log')
        plt.show()
    
    dy_N = np.diff(y, n=N) * mp.mpf(a)**mp.mpf(N/1.0)
    x = (x[1:] + x[:-1])/2
    N_mp = mp.mpf(N/1.0)
    sigma[N] = (dy_N**2).mean()**.5 / pi**N_mp * N_mp**(-N_mp)

fig, ax = plt.subplots()
plt.plot(sigma, "ro")
ax.set_yscale('log')
plt.show()