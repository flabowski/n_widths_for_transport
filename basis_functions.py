# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:24:29 2022

@author: florianma
"""
import numpy as np
import scipy as sc
from scipy.linalg import svd, orth, qr
import matplotlib.pyplot as plt
cmap = plt.cm.plasma
import timeit


def eigh_SVD(X):
    m, n = X.shape
    if m > n:
        S, V = sc.linalg.eigh(X.T@X)
        ll = S > 0
        S = S[ll]
        V = V[:, ll]
        S = S**.5
        U = np.matmul(X, V / S)
        VT = V.T
    else:
        S, U = sc.linalg.eigh(X@X.T)
        ll = S > 0
        S = S[ll]
        U = U[:, ll]
        S = S**.5
        VT = np.matmul(np.transpose(U) / S[:, None], X)
    return U[:, ::-1], S[::-1], VT[::-1, :]


def normalize(U):
    # U_n = U / (np.sum(U**2, axis=1)**.5)[:, None]
    U_n = U / (np.sum(U**2, axis=0)**.5)
    return U_n


def projection_error(X, basis):
    a, b = X.shape
    a, r = basis.shape
    if a <= r:
        difference = X - (basis @ basis.T) @ X  # (a, a) x a, b
        # ~ a*r*a + a*a*b
    else:  # r < a:
        difference = X - basis @ (basis.T @ X)  # a, r x (r, b)
        # ~ r*a*b + a*r*b
    return difference


def L2_per_snapshot(difference):
    L2 = np.mean(difference**2, axis=0)**.5
    return L2


def L2(difference):
    return np.mean(difference**2)**.5


def integrate(x):
    # assumes a-b = 1
    return np.mean(x)


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
        print("calc_error:", M, N, r_max)
        delta_n = np.zeros((r_max,), dtype=np.float64)
        d_n = np.zeros((r_max,), dtype=np.float64)
        dr = int(r_max/200)*2  # 100 samples, only even r
        for r in range(2, r_max, dr):
            print(r, end=", ")
            U_r = self.reduced_basis(rank=r)

            # fig, ax = plt.subplots()
            # plt.imshow(U_r.T @ U_r, interpolation="nearest")
            # plt.show()

            diff = projection_error(X_test, U_r)
            norm = L2_per_snapshot(diff)
            # norm = np.mean((X - basis @ (basis.T @ X))**2)**.5
            delta_n[r] = integrate(norm)
            d_n[r] = norm.max()
            if r in []: #[2, 4, 6, 8, 10, 12 ,52, 952]:
                fig, ax = plt.subplots()
                ax.plot(norm, "b.", ms=1)
                ax.plot(delta_n[r]*np.ones_like(norm), "g--")
                ax.plot(d_n[r]*np.ones_like(norm), "r--")
                #ax.set_ylim(0, norm.max()*1.2)
                plt.show()
        return delta_n, d_n

    def calc_error_analytic(self, X):
        # https://stats.stackexchange.com/a/211693
        a, b = X.shape
        VT = self.U.T @ X  # (r, m) x (m, n)
        S = np.sum(VT**2, axis=1)**.5  # (r, )
        # fig, ax = plt.subplots()
        # plt.plot(S)
        # ax.set_yscale('log')
        # #plt.show()
        delta_n = (np.cumsum(S[::-1]**2)[::-1]/a/b)**.5
        return delta_n, S


class SVD(Basis):
    name = "svd"

    def __init__(self, X, fast=False):
        # based on snapshots X
        self.X = X
        if fast:
            U, S, VT = eigh_SVD(X)
        else:
            U, S, VT = np.linalg.svd(X, full_matrices=False)
        # is_flipped = VT[:, 1] < 0
        # U[:, is_flipped] *= -1
        # VT[is_flipped, :] *= -1
        is_flipped = U[1, :] < 0
        U[:, is_flipped] *= -1
        VT[is_flipped, :] *= -1
        self.U = U
        self.S = S
        self.VT = VT
        return None

    def calc_error(self, X_test=None, r_max=None, test=None):
        m, n = self.U.shape[0], self.VT.shape[1]
        if not isinstance(X_test, np.ndarray):
            X_test = self.X
            delta_n = (np.cumsum(self.S[::-1]**2)[::-1]/m/n)**.5
            delta_n = delta_n[:r_max]
            d_N = np.zeros_like(delta_n)
            return delta_n, d_N
        else:
            return super().calc_error(X_test, r_max)

class Trigonometric(Basis):
    name = "trigonometric"

    def __init__(self, x, fun=np.sin, r=None):
        # based on analytic sine functions
        m = x.size
        if not r:
            r = m
        U = np.ones((m, r+1))
        delta_x = x[1] - x[0]
        T = x[-1]-x[0]+delta_x
        A = 2*delta_x**0.5 * np.sin(np.pi/4)
        A = (2*delta_x)**.5
        for i in range(r):
            omega = 2 * np.pi/(4*T) * (2*i+1)
            U[:, i+1] = A * fun(omega*x)
        self.U = normalize(U)
        return

    def sort(self, X, PLOT=True):
        m,n = self.shape
        V_k = self.U.reshape(m, -1, 2)  # x, r, 2
        V_1 = V_k[:, :, 0].copy()  # sin
        c = V_1.T @ X  # V_1.T @ X is slow!
        c_n = (c**2).sum(axis=1)
        order = np.argsort(-c_n)
        if PLOT:
            V_1 = V_k[:, :, 1].copy()  # sin
            # c2 = V_1.T @ X  # V_1.T @ X is slow!
            c_n2 = (c**2).sum(axis=1)
            fig, ax = plt.subplots()
            ax.plot(self.frequencies, c_n**.5, ".", ms=1)
            ax.plot(self.frequencies, c_n2**.5, ".", ms=1)
            ax.set_yscale('log')
            plt.show()
        self.U = V_k[:, order, :].reshape(m, -1)
        self.frequencies = self.frequencies[order]
        print("sorted.")
        return None


class Fourier(Basis):
    name = "Fourier"
    def __init__(self, x, r=None):
        a = x.size
        if not r:
            r = a
        U = np.ones((a, r*2))
        delta_x = x[1] - x[0]
        T = x[-1]-x[0]+delta_x
        # Nyquist–Shannon sampling theorem
        omega_max = 1/2 * 2*np.pi * a/T
        for i in range(1, r):
            omega = 2 * np.pi * i / T
            if omega <= omega_max:
                U[:, 2*i-1] = np.sin(omega*x)
                U[:, 2*i] = np.cos(omega*x)
            else:
                print("Omega max!", i, omega, omega_max)
                U = U[:, :2*(i-1)]
                break
        self.U = normalize(U)
        return


class Trigonometric2(Basis):
    name = "trigonometric2"

    def __init__(self, x, fun=np.sin, r=None):
        # based on analytic sine functions
        m = x.size
        if not r:
            r = m
        U = np.ones((m, r))
        delta_x = x[1] - x[0]
        A = 2*delta_x**0.5 * np.sin(np.pi/4)
        A = (2*delta_x)**.5
        T = x[-1]-x[0]+delta_x
        a = x.size
        # Nyquist–Shannon sampling theorem
        omega_max = 1/2 * 2*np.pi * a/T
        for i in range(r):
            # omega = i*np.pi
            omega = np.pi * (2*i+1)
            if omega <= omega_max:
                U[:, i] = A * np.sin(omega*x)
                # U[:, 2*i+2] = A * np.sin(omega*x)
                # U[:, 2*i+3] = A * np.cos(omega*x)
            else:
                print(i, omega, omega_max)
                U = U[:, :i]
                # U = U[:, :2*i+2]
                break
            # U[:, i+r] = A * np.cos(omega*x)
            # if i == 0:
            #     U[:, i] = delta_x**.5
        self.U = normalize(U)
        return


class TrigonometricOdd(Basis):
    name = "trigonometric_all"

    def __init__(self, x, fun=np.sin, r=None):
        m = x.size
        if not r:
            r = m
        delta_x = x[1] - x[0]
        A = (2*delta_x)**.5  # =2*delta_x**0.5 * np.sin(np.pi/4)
        T = x[-1]-x[0]+delta_x
        N = x.size
        i = np.arange(r)
        f = (2*i+1)/2
        omega = f * 2*np.pi
        omega_max = 1/2 * 2*np.pi * N/T
        nyquist_shannon_sampling_theorem = omega <= omega_max
        omega = omega[nyquist_shannon_sampling_theorem]
        U = np.concatenate((A * np.sin(omega[None, :, None]*x[:, None, None]),
                            A * np.cos(omega[None, :, None]*x[:, None, None])),
                            axis=2).reshape(m, -2)
        self.U = normalize(U)
        self.frequencies = f[nyquist_shannon_sampling_theorem]
        self.A = A
        self.delta_x = delta_x
        self.x = x
        return


class Greedy(Basis):
    name = "greedy"

    def __init__(self, X, r=None):
        # based on snapshots X
        print("generating greedy basis. rank: ")
        m, n = X.shape
        if not r:
            r = min(m, n)
        U = np.zeros_like(X[:, 0, None], dtype=np.float64)
        err = np.zeros(r, dtype=np.float64)
        for i in range(r):
            print(i, end=", ")
            psi_i, err[i] = self.next_basis_vec(X, U)
            U = np.c_[U, psi_i]
        print()
        Q, R = qr(U, mode="economic")
        self.U = Q
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
        X = np.zeros((m, n), dtype=np.float64)  # snapshot matrix
        for j, mu_j in enumerate(mu):
            X[:, j] = self.u(x, mu_j)
        W = (X[:, :-1]+X[:, 1:])/2
        Q, R = qr(W, mode="economic")
        self.U = Q
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
        U = np.zeros((m, r), dtype=np.float64)
        for i in range(r):
            is_one = (i/r <= x) & (x < (i+1)/r)
            U[is_one, i] = 1.0
        assert np.all(np.sum(U, axis=1) == 1), "not orthogonal"
        return normalize(U)


class Sinc(Basis):
    name = "sinc"

    def __init__(self, domain):
        self.domain = domain

    def reduced_basis(self, rank):
        x = self.domain()
        r = rank
        m = x.size
        U = np.zeros((m, r), dtype=np.float64)
        knots = np.linspace(0, 1, r)
        if rank == 1:
            dk = 1
        else:
            dk = knots[1]-knots[0]
        for i, _x_ in enumerate(knots):
            U[:, i] = np.sinc((x-_x_)/dk)
        Q, R = qr(U, mode="economic")
        return Q


if __name__ == "__main__":
    plt.close("all")
    from initial_conditions import Domain, Heaviside
    N = 100
    x = Domain([-1, 1], N)
    dx = 2/N
    # x.x = np.linspace(-1, 1, N)
    trig_basis = TrigonometricOdd(x())
    U = trig_basis.U#[:, :5]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(U.T@U)
    ax2.imshow(U@U.T)
    
    
    # xf = Domain([-1, 1], 100*N)
    # trig_basis = Trigonometric2(xf)
    # U_fine = trig_basis.U#[:, :5]
    
    # fig, ax = plt.subplots()
    # for i in range(N):
    #     omega = 2 * np.pi/(4) * (2*i+1)
    #     omega = np.pi * (2*i+1)
    #     f = 1 * np.sin(omega*x())
    #     ff = 1 * np.sin(omega*xf())
    #     plt.plot(x(), f, "ro", label="i="+str(i), color=cmap(i/(N-1)))
    #     plt.plot(xf(), ff, "--", color=cmap(i/(N-1)))
    # plt.legend()
    # plt.show()
    asd
    
    x = Domain([0, 1], 100)
    mu = Domain([0, 1], 100)
    u = Heaviside()
    X = u(x(), mu())
    svd_basis = SVD(X)
    sinc_basis = Sinc(x)
    trig_basis = Trigonometric(x)
    greedy_basis = Greedy(X)
    lna_basis = LNA(u, x)
    lpf_basis = LPF(x)
    X_test = X
    U = sinc_basis.reduced_basis(10)
    fig, ax = plt.subplots()
    plt.imshow(U, interpolation="nearest")
    asd
    for basis in [sinc_basis]:   # , svd_basis, trig_basis, greedy_basis, lna_basis, lpf_basis]:
        basis.calc_error(X_test)
