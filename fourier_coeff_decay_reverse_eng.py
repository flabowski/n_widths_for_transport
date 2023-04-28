# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:13:36 2022

@author: florianma
"""


import numpy as np
import matplotlib.pyplot as plt
plt.close("all")


def plot_snapshots_lines(x, X):
    m, n = X.shape
    fig, ax = plt.subplots()
    for i in range(n):
        if i % 10 == 0:
            ax.plot(x, X[:, i], "o--")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    plt.show()


M = 100  # samples
T = 1  # domain size
t = np.linspace(0, T, M, endpoint=False)  # dolfin: 1 ... 0
dx = T/M  # = x[1] - x[0]
U = np.zeros((M, M))
VT = np.zeros((M, M))
for i in range(M):
    # A, w, p = guess(i, dx)
    A = 1
    w = i * 2 * np.pi
    p = 0
    U[:, i] = A * np.sin(w*t + p)
    VT.T[:, i] = A * np.cos(w*t + p)


def normalize(U):
    U_n = U / np.sum(U**2, axis=0)**.5
    return U_n


S_ = 1 / np.arange(1, M+1)**2
S_ = np.exp(-np.arange(1, M+1))
X = U * S_ @ VT
# X = normalize(X)
# X = normalize(X.T).T
plt.imshow(X)
plt.show()
fig, ax = plt.subplots()
plt.imshow(X)
plt.show()
plot_snapshots_lines(t, X)
cp = X@X.T
cp
