# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 10:11:45 2022

@author: florianma
"""
import numpy as np
import matplotlib.pyplot as plt
from basis_functions import normalize, projection_error, L2_per_snapshot, L2, integrate


def u(x, mu):
    y = np.zeros_like(x)
    y[x < mu] = 0.0
    y[x == mu] = .5
    y[x > mu] = 1.0
    return y


def getX(M, N, random=False):
    dx, dm = 1/M, 1/N
    x = np.linspace(dx/2, 1-dx/2, M)
    mu = np.linspace(dm/2, 1-dm/2, N)
    if random:
        mu = np.random.rand(N,)
    X = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            X[i, j] = u(x[i], mu[j])
    return X


a, b, n = 500, 500, 500
X = getX(a, b)
U, S, VT = np.linalg.svd(X, full_matrices=False)
# plt.plot(S)
# plt.gca().set_yscale('log')
# plt.show()
# asd

err_S = (np.cumsum(S[::-1]**2)[::-1]**.5)[:n]
d_n = np.zeros((n,))
# X_test = getX(a, b, random=True)
X_test = X
for N in range(1, n+1):
    U_, S_, VT_ = U[:, :N], S[:N], VT[:N, :]
    difference = X_test - (U_ @ U_.T) @ X_test
    L2_norm2 = np.sum(difference**2)
    SVD_error = np.sum(S[N:]**2)  # == L2_norm**2. Depends on a and b !!!
    print(N, L2_norm2, SVD_error)
    norm = np.mean(difference**2, axis=0)**.5
    d_n[N-1] = integrate(norm)


i = np.arange(1, n+1)
estimate = (1/2 - 4/np.pi**2 * np.cumsum(1/(2*i - 1)**2)) ** 0.5

# plt.plot(i, err_S, "r.")
plt.plot(i, estimate, "k--")
plt.plot(i-1, d_n, "b.")
plt.gca().set_yscale('log')
plt.show()


# cmap = plt.cm.plasma
# dx, dm = 1/a, 1/b
# x = np.linspace(dx/2, 1-dx/2, a)
# fig, ax = plt.subplots()
# for i in range(5):
#     A = (2*dx)**.5
#     omega = 2 * np.pi/(4*1) * (2*i+1)
#     s = A * np.sin(omega*x)
#     print(U[:, i]/s)
#     plt.plot(x, U[:, i], "o", ms=1, color=cmap(i/4), label="mode {:.0f}".format(i))
#     ax.plot(x, s, "k--", lw=.5)# label="mode {:.0f}".format(i))
# ax.plot([-2, -1], [0, 0], "k--", lw=.5, label="theoretical results")
# plt.legend(prop={'size': 6})
# plt.grid(which="both")
# plt.xlim([0, 1])
# plt.ylim([-0.035, 0.035])
# plt.show()