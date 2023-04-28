# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:21:26 2022

@author: florianma
"""
import numpy as np
from scipy.linalg import svd, orth, qr
from n_width import get_snapshot_matrix, u0, normalize
from basis_functions import Basis, Sinc
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)


if __name__ == "__main__":
    m, n, r = 1000, 1000, 11

    from initial_conditions import Domain, Heaviside
    plt.close("all")
    x = Domain([0, 1], m)
    mu = Domain([0, 1], n)
    u = Heaviside()
    X = u(x(), mu())
    trials = 1001
    error = np.zeros(trials)
    eb = np.linspace(0, .5, trials)
    orthogonalize = True
    for j, extend_by in enumerate(eb):
        # extend_by = 0.2
        T = 1+extend_by
        dk = T/(r-1)

        knots = np.linspace(-extend_by/2, 1+extend_by/2, r)
        U = np.zeros((m, r), dtype=np.float64)
        omega = np.pi/dk
        for i, _x_ in enumerate(knots):
            f1 = np.sinc((x()-_x_)/dk)
            # np.sin(omega*(x()-_x_))/(x()-_x_)/omega
            f1 = normalize(f1)
            U[:, i] = f1
        if orthogonalize:
            Q, R = qr(U, mode="economic")
            basis = Q
        else:
            basis = U
        C = (basis.T @ X)
        X_ = basis @ C
        error[j] = np.mean((X-X_)**2)
        print(extend_by, error[j])
    fig, ax = plt.subplots()
    plt.plot(eb, error, "r.")
    plt.show()

    extend_by = eb[np.argmin(error)]
    T = 1+extend_by
    dk = T/(r-1)

    knots = np.linspace(-extend_by/2, 1+extend_by/2, r)
    U = np.zeros((m, r), dtype=np.float64)
    omega = np.pi/dk
    for i, _x_ in enumerate(knots):
        f1 = np.sinc((x()-_x_)/dk)
        f1[x() == _x_] = 1.0
        f1 = normalize(f1)
        U[:, i] = f1

    if orthogonalize:
        Q, R = qr(U, mode="economic")
        basis = Q
    else:
        basis = U
    C = (basis.T @ X)
    X_ = basis @ C
    fig, ax = plt.subplots()
    for i in range(r):
        plt.plot(x(), U[:, i])
    plt.show()
    fig, ax = plt.subplots()
    for i in range(r):
        plt.plot(x(), basis[:, i])
    plt.show()
    
        # fig, ax = plt.subplots()
        # plt.plot(x(), X[:, 52])
        # plt.plot(x(), X_[:, 52])
        # plt.show()

        # fig, ax = plt.subplots()
        # for j in range(r):
        #     plt.plot(x(), U[:, j])
        # plt.plot(x(), np.sum(U, axis=1), "r--")
        # plt.show()
    # asd
    for j in range(1, n//1, n//10):
        fig, ax = plt.subplots()
        plt.plot(x(), X[:, j])
        plt.plot(x(), X_[:, j])
        plt.show()
    # fig, ax = plt.subplots()
    # plt.imshow(U @ U.T, interpolation="nearest")
    # plt.show()
    # fig, ax = plt.subplots()
    # plt.imshow(U.T @ U, interpolation="nearest")
    # plt.show()
