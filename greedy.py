# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 12:42:39 2022

@author: florianma
"""
import numpy as np
from n_width import get_snapshot_matrix, u2, normalize
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


def next_basis_vec(X, U):
    error = X - U @ (U.T @ X)
    L2 = np.mean(error**2, axis=0)**.5
    worst = np.argmax(L2)
    m, r = U.shape
    print(m, r)
    if r < 10:
        fig, (ax, ax2) = plt.subplots(2)
        ax.plot(mu, L2, "b.")
        ax.plot(mu[worst], L2[worst], "r.")
        ax.set_xlabel("mu")
        ax.set_ylabel("L2 error")
        ax.set_xlim([0, 1])
        # plt.show()
        # fig, ax = plt.subplots()
        ax2.plot(np.linspace(0, 1, m), normalize(error[:, worst]))
        plt.tight_layout()
        ax2.set_xlabel("x")
        ax2.set_ylabel("approximation error")
        ax2.set_xlim([0, 1])
        plt.show()
    return normalize(error[:, worst]), L2[worst]


def greedy_basis(X, r=None):
    m, n = X.shape
    if not r:
        r = min(m, n)
    U = np.zeros_like(X[:, 0, None])
    err = np.zeros(r,)
    for i in range(r):
        psi_i, err[i] = next_basis_vec(X, U)
        U = np.c_[U, psi_i]
    fig, ax = plt.subplots()
    plt.plot(err, "b.")
    ax.set_xlabel("iteration")
    ax.set_ylabel("max. error")
    plt.xlim([0, r])


if __name__ == "__main__":
    M = 1000  # samples
    T = 1  # domain size
    epsilon = 0.05  # jump size / 2

    x = np.linspace(0, T, M, endpoint=False)  # dolfin: 1 ... 0
    dx = T/M
    mu = np.arange(0+0*dx, T+0*dx, dx)  # x  # parameter space
    def u_(x, t): return u2(x, t, epsilon)
    X = get_snapshot_matrix(x, mu, u_)
    U = greedy_basis(X, r=100)
    asd
    # U = normalize(X.mean(axis=1)[:, None])
    U = normalize(X[:, 208, None])  # 0 0.428; 1 0.4265
    # U = normalize(X[:, 0, None])
    # from scipy.linalg import svd
    # U, S, VT = svd(X, full_matrices=False)
    # U = U[:, 0, None]
    # U = np.ones_like(X[:, 0, None])*0.5
    # U = np.zeros_like(X[:, 0, None])  # 0 0.9927; 1 0.4844; 2 0.32952

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 1, M), U[:, 0])
    plt.show()

    r = 10
    err = np.zeros(r,)
    for i in range(r):
        psi_i, err[i] = next_basis_vec(X, U)
        print(i, err[i], sep="\t")
        U = np.c_[U, psi_i]
    fig, ax = plt.subplots()
    plt.plot(err, "b.")
    # worst_snapshot = X[largest_error(X, U)]
    # normalize(X[0])
