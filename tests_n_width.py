# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 20:22:33 2022

@author: florianma
"""
from n_width import (u1, get_snapshot_matrix, singular_values, fit_and_plot,
                     sinfunc, cosfunc, sin_basis, greedy_basis, truncate,
                     calc_error, local_neighbours_average, get_large_step_matrix,
                     constantins_basis_paper, extend_to_2pi, u2, sigmoid, fit)
from scipy.optimize import curve_fit
from scipy.linalg import svd, orth, qr
import numpy as np
import matplotlib.pyplot as plt
cmap = plt.cm.plasma
plt.close("all")


def rms(difference):
    L2 = np.mean(difference**2, axis=0)**.5
    return L2


def projection_error(X, basis):
    difference = X - (basis @ basis.T) @ X
    return rms(difference)


def decompose_large_X():
    M = 100
    T = 1
    dx = T/M
    X = get_large_step_matrix(M)
    x = np.linspace(dx/2, 4-dx/2, 4*M)
    mu = np.linspace(dx/2, 4-dx/2, 4*M)
    fig, ax = plt.subplots()
    plt.imshow(X)
    plt.show()
    U4, S4, VT4 = singular_values(X)
    fit_and_plot(x, U4, 10, sinfunc)
    fit_and_plot(mu, VT4.T, 10, cosfunc)

    epsilon = 0.0000025  # jump size / 2
    x_ = np.linspace(dx/2, T-dx/2, M)  # dolfin: 1 ... 0
    mu_ = x_  # [1:-1][::2]  # np.linspace(T/N/2, T-T/N/2, N, endpoint=True)
    N = len(mu)
    def u_(x_, t): return u1(x_, t, epsilon)
    X = get_snapshot_matrix(x_, mu_, u_)
    U, S, VT = singular_values(X)
    X2, U2, S2, VT2 = extend_to_2pi(x_, mu_, X, U, S, VT)

    for i in range(10):
        fig, ax = plt.subplots()
        ax.plot(x, U4[:, i]*2, "g--")
        ax.plot(x_, U[:, i], "b.")
        ax.plot(x, U2[:, i], "r.")


def max_error_jump():
    M = 100  # samples, 13, 63, 313, 1563, 7813
    # assert M % 2 == 1
    # N = 500
    T = 1  # domain size
    epsilon = 0.0000025  # jump size / 2
    dx = T/M
    # x = np.linspace(0, T+dx/2, M, endpoint=False)  # dolfin: 1 ... 0
    x = np.linspace(dx/2, T-dx/2, M)  # dolfin: 1 ... 0
    # dx = T/M  # = x[1] - x[0]
    mu = x  # [1:-1][::2]  # np.linspace(T/N/2, T-T/N/2, N, endpoint=True)
    N = len(mu)
    # mu = np.linspace(0, T, M+1)
    # a = fit_sigmoid(epsilon)
    # def u_(x, t): return sigmoid(x, t, a)
    def u_(x, t): return u1(x, t, epsilon)
    X = get_snapshot_matrix(x, mu, u_)

    # X[0, 0] = 0

    fig, ax = plt.subplots()
    plt.plot(x, u_(x, .5), "o-")
    ax.set_xlabel("x")
    ax.set_xlabel("u(x)")
    plt.show()
    print(X)

    U, S, VT = singular_values(X)
    # plt.plot(1*(S*np.arange(1, M+1)), ".")
    # plt.show()

    X2, U2, S2, VT2 = extend_to_2pi(x, mu, X, U, S, VT)
    # frequenz_analyse(U)  # needs high sampling rate

    fit_and_plot(x, U, 10, sinfunc)
    fit_and_plot(mu, VT.T, 10, cosfunc)
    # asd
    U_guess = sin_basis(x, 1000)
    U_greedy = greedy_basis(X)

    # N = M*10
    mu_test = np.random.uniform(0, T, N)  # monte carlo
    mu_test = np.linspace(0, T, N+1)  # monte carlo
    X_test = get_snapshot_matrix(x, mu_test, u_)

    M, R = U_guess.shape
    def rb(r): return truncate(U_guess, r)
    rms_error0, max_error0 = calc_error(rb, X_test)

    M, R = U_guess.shape
    def rb(r): return truncate(U_greedy, r)
    rms_error1, max_error1 = calc_error(rb, X_test)

    M, R = U.shape
    def rb(r): return truncate(U, r)
    rms_error2, max_error2 = calc_error(rb, X_test)

    # N = M
    x_test = np.linspace(0, T, M, endpoint=False)
    mu_test = np.random.uniform(0, T, N)  # monte carlo
    mu_test = np.linspace(0, T, N)  # monte carlo
    X_test = get_snapshot_matrix(x_test, mu_test, u_)

    M, R = U_guess.shape
    def rb(r): return local_neighbours_average(x_test, u_, r)
    rms_error3, max_error3 = calc_error(rb, X_test)

    M, R = U_guess.shape
    def rb(r): return constantins_basis_paper(x_test, r)
    rms_error4, max_error4 = calc_error(rb, X_test)
    # fig, ax = plt.subplots()
    # plt.plot(x_test, X_test[:, 500], "g--", marker=".")  # mu
    # plt.plot(x, X[:, 50], "r--", marker=".")
    # plt.show()

    n = np.arange(1, len(max_error3)+1)
    fig, ax = plt.subplots()
    ax.plot(max_error0, ".", label="max_error sin basis")
    ax.plot(max_error1, ".", label="max_error greedy basis")
    ax.plot(max_error2, ".", label="max_error SVD basis")
    ax.plot(max_error3, ".", label="max_error Psi")
    ax.plot(max_error4, ".", label="max_error lna")
    ax.plot(n, 1/(2*n**.5), "k--", label="1/(2sqrt(n)")
    ax.plot(n, 1/(4*n**.5), "k--", label="1/(4sqrt(n)")
    ax.set_yscale('log')
    plt.legend()
    ax.set_xlim(0, len(max_error3)/2)
    plt.ylim(1/100, 1)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(rms_error0, ".", label="rms_error sin basi")
    ax.plot(rms_error1, ".", label="rms_error greedy basis")
    ax.plot(rms_error2, ".", label="rms_error SVD basis")
    ax.plot(rms_error3, ".", label="rms_error Psi")
    ax.plot(rms_error4, ".", label="rms_error lna")
    ax.plot(n, 1/(2*n**.5), "k--", label="1/(2sqrt(n))")
    ax.plot(n, 1/(4*n**.5), "k--", label="1/(4sqrt(n))")
    ax.set_yscale('log')
    plt.legend()
    ax.set_xlim(0, len(max_error3)/2)
    plt.ylim(1/100, 1)
    plt.show()

    extend_to_2pi(x, mu, X, U, S, VT)

    bv = local_neighbours_average(x, u_, 200)
    fig, ax = plt.subplots()
    ax.imshow(bv.T @ bv)

    mu = np.linspace(0, 1, 100+1)
    U = get_snapshot_matrix(x, mu, u_)
    W = (U[:, :-1]+U[:, 1:])/2
    # GS = orth2(W)
    Q, R = qr(W, mode="economic")
    fig, ax = plt.subplots()
    ax.imshow(Q)
    return


def compare_dn():
    M = 100  # samples
    T = 1  # domain size
    epsilon = 0.0000000005  # jump size / 2

    x = np.linspace(0, T, M, endpoint=False)  # dolfin: 1 ... 0
    dx = T/M  # = x[1] - x[0]
    mu = np.arange(0+0.0*dx, T-0.0*dx, dx)  # x  # parameter space
    # mu = np.linspace(0, T, M+1)
    def u_(x, t): return u1(x, t, epsilon)
    X = get_snapshot_matrix(x, mu, u_)
    U, S1, VT = singular_values(X)

    def u_(x, t): return u2(x, t, epsilon)
    X = get_snapshot_matrix(x, mu, u_)
    U, S2, VT = singular_values(X)

    a = fit_sigmoid(epsilon)
    def u_(x, t): return sigmoid(x, t, a)
    X = get_snapshot_matrix(x, mu, u_)
    U, S3, VT = singular_values(X)

    fig, ax1 = plt.subplots()
    ax1.plot(x, u1(x, 0.5, epsilon))
    ax1.plot(x, u2(x, 0.5, epsilon))
    ax1.plot(x, sigmoid(x, 0.5, a))
    plt.show()

    fig, ax1 = plt.subplots()
    plt.plot(S1, ".", label="linear")
    plt.plot(S2, ".", label="piecewise quadratic")
    plt.plot(S3, ".", label="sigmoid")
    plt.legend()
    ax1.set_xlabel("order")
    ax1.set_ylabel("singular value")
    ax1.set_yscale('log')
    ax1.set_xlim([0, len(S1)])
    ax1.set_ylim([1e-10, 1e3])
    ax1.grid(which="both")
    plt.tight_layout()
    plt.show()


def frequenz_analyse(U):
    m, n = U.shape
    t = np.linspace(0, 1, m)
    for i in range(20):
        sp = np.fft.fft(U[:, i])
        freq = np.fft.fftfreq(t.shape[-1])
        positiv_axis = freq >= 0
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(t, U[:, i], ".")
        ax2.plot(freq[positiv_axis], sp.real[positiv_axis], ".")
        ax2.plot(freq[positiv_axis], sp.imag[positiv_axis], ".")
        plt.show()


def fit_sigmoid(eps):
    M = 200
    tt = np.linspace(0, 1, M)
    yy = u2(tt, 0.5, eps)
    popt, pcov = curve_fit(sigmoid, tt, yy, [0.5, 50])
    t, a = popt
    yy_hat = sigmoid(tt, t, a)
    fig, ax = plt.subplots()
    plt.plot(tt, yy, "g.")
    plt.plot(tt, yy_hat, "r.")
    plt.show()
    return a


def f(s1, e1, s2, e2, dx, epsilon):
    x = np.arange(s1-1e-8, e1+1e-8, dx)
    mu = np.arange(s2-1e-8, e2+1e-8, dx)
    def u_(x, t): return u1(x, t, epsilon)
    X = get_snapshot_matrix(x, mu, u_)
    U, S, VT = singular_values(X)
    p1 = fit(x, U, 5, sinfunc)
    p2 = fit(mu, VT.T, 5, cosfunc)
    print(x[0], x[-1], mu[0], mu[-1], p1, p2, sep="\t")
    return e1, e2


def find_sart_end():
    M = 1000  # samples
    T = 1  # domain size
    epsilon = 0.025  # jump size / 2

    # mu = np.linspace(0, 1, M, endpoint=False)  # dolfin: 1 ... 0
    # x = np.linspace(0, 1, M+1, endpoint=True)  # dolfin: 1 ... 0
    dx = T/M  # = x[1] - x[0]
    for s1 in [-dx, -dx/2, 0, dx/2, dx]:
        for e1 in [1-dx, 1-dx/2, 1, 1+dx/2, 1+dx]:
            for s2 in [-dx, -dx/2, 0, dx/2, dx]:
                for e2 in [1-dx, 1-dx/2, 1, 1+dx/2, 1+dx]:
                    for s1 in 0.01 + np.linspace(-dx, dx, 101):
                        er1, er2 = f(s1, e1,  s2, e2, dx, epsilon)
                    # print(s1, e1, s2, e2, er1, er2, sep="\t")
                    # asd
    s1, e1, s2, e2 = 0.0172, 1.0072, 0.005, 0.995

    # x = np.arange(s1, e1, dx)
    # mu = np.arange(s2, e2, dx)
    x = np.arange(s1-1e-8, e1+1e-8, dx)
    mu = np.arange(s2-1e-8, e2+1e-8, dx)
    def u_(x, t): return u1(x, t, epsilon)

    X = get_snapshot_matrix(x, mu, u_)
    U, S, VT = singular_values(X)
    e1 = fit_and_plot(x, U, 10, sinfunc)
    e2 = fit_and_plot(mu, VT.T, 10, cosfunc)
