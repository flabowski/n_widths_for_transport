# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:59:07 2022

@author: florianma
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd, orth, qr
from scipy.optimize import curve_fit
from numpy import sin, cos, pi
from initial_conditions import Domain, Heaviside, LinearRamp, SmoothRamp, Sigmoid
from basis_functions import SVD, Trigonometric, Greedy, LNA, LPF, normalize
import matplotlib.pyplot as plt
import matplotlib as mpl
cmap = plt.cm.plasma
plt.close("all")


if __name__ == "__main__":
    print("main")
    T = 1
    m, n, r = 2000, 1000, 500
    x = Domain([0, T], m)()
    mu = Domain([0, T], n)()
    u = Sigmoid(50)
    u.plot(x)
    X = u(x, mu)
    svd_basis = SVD(X)
    U = svd_basis.U
    S = svd_basis.S
    VT = svd_basis.VT
    M, N = X.shape
    m2, n2 = m*4, n*4
    dx2, dm2 = 4/m2, 4/n2
    x2 = np.linspace(dx2/2, 4-dx2/2, m*4)
    dx = x[1]-x[0]
    mu2 = np.linspace(dm2/2, 4-dm2/2, n*4)
    dm = mu[1]-mu[0]
    U2 = np.zeros((m*4, n))
    VT2 = np.zeros((n, n*4))

    for i in range(n):
        w1 = w2 = w = (i/2+.25) * 2 * np.pi
        U2[:, i] = normalize(np.sin(w*x2))
        VT2[i, :] = normalize(np.cos(w*mu2))
        if i < 25:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(x, U[:, i], "k.")
            ax1.plot(x2, U2[:, i], "r.")
            # ax2.plot(mu, VT[i, :], "k.")
            # ax2.plot(mu, VT2[i, :], "r.")
            # plt.show()
            # samplingFrequency = m
            # tpCount = len(U2[:, i])
            # values = np.arange(int(tpCount/2))
            # timePeriod = tpCount/samplingFrequency
            frequencies = np.arange(int(m/2))/T


            sp = np.fft.fft(U2[:, i])
            freq = np.fft.fftfreq(x.shape[-1])
            # fig, ax = plt.subplots()
            ax2.plot(1/frequencies, sp.real[:int(m/2)], "b.")
            ax2.plot(1/frequencies, sp.imag[:int(m/2)], "r.")
            plt.show()
            
    # SVT2 = U2.T @ X
    # S2 = np.sum(SVT2**2, axis=1)**.5
    # fig, ax = plt.subplots()
    # ax.plot(S2)
    # ax.plot(S)
    # ax.set_yscale("log")
    
    # X_approx = (U2[:, :15]*S2[:15]) @ VT2[:15, :]
    # fig, ax = plt.subplots()
    # ax.plot(x, X[:, 500])
    # ax.plot(x, X_approx[:, 500])
    
