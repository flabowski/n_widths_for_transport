# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:53:20 2023

@author: florianma
"""
import matplotlib.pyplot as plt
import numpy as np
plt.close("all")


def f(x):
    return np.sin(2*np.pi*x)


def f_(x):
    return 2*np.pi*np.cos(2*np.pi*x)


def u(x, v, t):
    return f(x-v*t)


def u_(x, v, t):
    return f_(x-v*t)


dt = 0.1
v = 1
x = np.linspace(0, 1, 1000)
u_0 = u(x, v, 0)
u_1 = u(x, v, 1*dt)
du_dt = (u_1 - u_0) / dt
u_mid = (u_(x, v, 0) + u_(x, v, 1*dt)) / 2
du_dx = v*u_mid

u_1 = u_0 - dt * v * du_dx

plt.figure()
plt.plot(x, u_0)
plt.plot(x, u_1)
plt.plot(x, u(x, v, 1*dt))
plt.show()


plt.figure()
plt.plot(x, du_dt)
plt.plot(x, du_dx)
plt.plot(x, du_dt+du_dx)
plt.show()