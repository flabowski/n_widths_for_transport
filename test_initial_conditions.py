# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:59:23 2023

@author: florianma
"""
import numpy as np
import matplotlib.pyplot as plt
from initial_conditions import CkRamp, Polynom_C1

def test_CkRamp():
    x = np.linspace(-1, 1, 2000)  # whole domain
    for k in range(6):
        u = CkRamp(0.5, k)
        plt.figure()
        plt.plot(x, u.half_wave_odd(x, 0.0))
        plt.show()


def test_Polynom_C1():
    x = np.linspace(-1, 1, 2000)  # whole domain
    u_1 = Polynom_C1()
    u_1.plot_q()
    fig, ax = plt.subplots()
    for mu in [0.0, 0.25, 0.5, 0.75, 1.0]:
        u_1.plot_u(mu, ax=ax, label=r'$u(\mu={:.2f}; x)$'.format(mu))
    plt.legend()
    plt.show()
    # for k in range(6):
    #     u = Polynoms(0.5, k)
    #     u.u(x, 0.0)
    #     plt.figure()
    #     plt.plot(x, u(x, 0.0))
    #     plt.show()
    


if __name__ == "__main__":
    # test_CkRamp()
    test_Polynoms()
