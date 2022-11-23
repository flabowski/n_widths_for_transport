# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 09:46:55 2022

@author: florianma
"""
import numpy as np
import matplotlib as mpl
from basis_functions import normalize, projection_error, L2_per_snapshot, L2, integrate, Trigonometric
from initial_conditions import Domain, Heaviside
import matplotlib.pyplot as plt

page_width_pt = 455.24
pt2in = 0.01389
pt2cm = 0.0352777778
cm2in = 1/2.54
plot_width_in = page_width_pt*pt2in/2
print(plot_width_in/cm2in)

plt.rcParams["figure.figsize"] = (plot_width_in, plot_width_in/1.61803398875)
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
mpl.rc('font', family='serif', size=12, serif='Computer Modern Roman')

results_path = "C:/Users/florianma/Dropbox/Kol-N-width-Oslo-Ulm/n_widths_for_transport/results/"

for a in (500, 1000, 2000, 5000, 10000, 20000):
    # a, b, N = 10000, 10000, 500
    N = 500
    b = a

    x = Domain([0, 1], a)
    mu = Domain([0, 1], b)
    u = Heaviside()
    X = u(x(), mu())
    U = Trigonometric(x, np.sin, N)
    X_test = X[:, ::10]
    delta_n, d_n = U.calc_error(X, N)
    np.save(results_path+"_delta_n_trigonometric_heaviside"+str(a)+".npy", delta_n)

    i = np.arange(1, N+1)
    plt.plot(i-1, delta_n, marker=".", ls="None", label="a={:.0f}".format(a))
    # plt.plot(i-1, d_n, "r.")
estimate = (1/2 - 4/np.pi**2 * np.cumsum(1/(2*i - 1)**2)) ** 0.5




# plt.plot(i, err_S, "r.")

plt.plot(i, estimate, "k--")
plt.gca().set_yscale('log')
plt.legend()
plt.show()
