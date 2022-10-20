# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:24:29 2022

@author: florianma
"""
import numpy as np
from scipy.linalg import svd, orth, qr
import matplotlib.pyplot as plt


def normalize(U):
    U_n = U / np.sum(U**2, axis=0)**.5
    return U_n


def rms(difference):
    L2 = np.mean(difference**2, axis=0)**.5
    return L2


def projection_error(X, basis):
    difference = X - (basis @ basis.T) @ X
    return rms(difference)


class Basis:
    def __init__(self):
        return

    def reduced_basis(self, rank):
        # truncation
        return self.U[:, :rank]

    def calc_error(self, X_test, r_max=None):
        M, N = X_test.shape
        if not r_max:
            r_max = N if N < M else M
        rms_error = np.zeros((r_max,))
        max_error = np.zeros((r_max,))
        for r in range(1, r_max//1, r_max//100):
            U_r = self.reduced_basis(rank=r)
            L2 = projection_error(X_test, U_r)
            rms_error[r] = np.sqrt(np.mean(L2**2))
            max_error[r] = L2.max()
        return rms_error, max_error


class SVD(Basis):
    name = "svd"

    def __init__(self, X):
        # based on snapshots X
        U, S, VT = np.linalg.svd(X, full_matrices=False)
        # is_flipped = VT[:, 1] < 0
        # U[:, is_flipped] *= -1
        # VT[is_flipped, :] *= -1
        is_flipped = U[0, :] < 0
        U[:, is_flipped] *= -1
        VT[is_flipped, :] *= -1
        self.U = U
        self.S = S
        self.VT = VT
        return None


class Trigonometric(Basis):
    name = "trigonometric"

    def __init__(self, domain, fun=np.sin, r=None):
        # based on analytic sine functions
        m = domain.size
        if not r:
            r = m
        U = np.zeros((m, r))
        T = domain.max-domain.min
        delta_x = domain.delta_x
        A = 2*delta_x**0.5 * np.sin(np.pi/4)
        x = domain()
        for i in range(r):
            omega = 2 * np.pi/(4*T) * (2*i+1)
            U[:, i] = A * fun(omega*x)
        self.U = normalize(U)
        return


class Greedy(Basis):
    name = "greedy"

    def __init__(self, X, r=None):
        # based on snapshots X
        print("generating greedy basis. rank: ")
        m, n = X.shape
        if not r:
            r = min(m, n)
        U = np.zeros_like(X[:, 0, None])
        err = np.zeros(r,)
        for i in range(r):
            print(i, end=", ")
            psi_i, err[i] = self.next_basis_vec(X, U)
            U = np.c_[U, psi_i]
        print()
        self.U = U
        return

    def next_basis_vec(self, X, U):
        error = X - U @ (U.T @ X)
        L2 = np.mean(error**2, axis=0)**.5
        worst = np.argmax(L2)
        return normalize(error[:, worst]), L2[worst]


class LNA(Basis):
    name = "local neighbours average"

    def __init__(self, u, domain):
        # based on the snapshot generating function u
        self.u = u
        self.domain = domain

    def reduced_basis(self, rank):
        x = self.domain()
        r = rank
        m, n = x.size, r+1
        dm = 1/n
        mu = np.linspace(dm/2, 1-dm/2, n)
        X = np.zeros((m, n))  # snapshot matrix
        for j, mu_j in enumerate(mu):
            X[:, j] = self.u(x, mu_j)
        W = (X[:, :-1]+X[:, 1:])/2
        Q, R = qr(W, mode="economic")
        self.basis = Q
        return Q


class LPF(Basis):
    name = "local pulse functions"

    def __init__(self, domain):
        # based on the snapshot generating function u
        self.domain = domain

    def reduced_basis(self, rank):
        x = self.domain()
        r = rank
        m = x.size
        U = np.zeros((m, r))
        for i in range(r):
            is_one = (i/r <= x) & (x < (i+1)/r)
            U[is_one, i] = 1.0
        assert np.all(np.sum(U, axis=1) == 1), "not orthogonal"
        return normalize(U)


if __name__ == "__main__":
    from test_functions import Domain, Heaviside
    plt.close("all")
    x = Domain([0, 1], 100)
    mu = Domain([0, 1], 100)
    u = Heaviside()
    X = u(x(), mu())
    svd_basis = SVD(X)
    trig_basis = Trigonometric(x)
    greedy_basis = Greedy(X)
    lna_basis = LNA(u, x)
    lpf_basis = LPF(x)
    X_test = X
    for basis in [svd_basis, trig_basis, greedy_basis, lna_basis, lpf_basis]:
        basis.calc_error(X_test)
