# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:12:27 2023

@author: florianma
"""

import time
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


b = 3  # #snapshots
r = 100
x = sp.Symbol('x')
y = sp.Symbol('y')
mu = sp.Symbol('mu')

gn_l = -1
gn_r = 1
for io in range(6):
    # print(sp.integrate(gn_l, x))
    # print(sp.integrate(gn_r, x))
    # print(sp.integrate(gn_l, (y, 0, x)), "|", 1/2 * sp.integrate(gn_r, (y, 0, 1)))
    a = sp.integrate(gn_l, (x, 0, x)) - sp.Rational(1, 2) * sp.integrate(gn_r, (x, 0, 1))
    b = sp.integrate(gn_r, (x, 0, x)) - sp.Rational(1, 2) * sp.integrate(gn_r, (x, 0, 1))
    gn_l, gn_r = sp.simplify(a), sp.simplify(b)
    print(sp.simplify(gn_l))
    print(sp.simplify(gn_r))
# asd
def f0(x):
    x = np.array(x)
    y = np.empty_like(x)
    y[x<0] = - x[x<0] - 1/2
    y[x>=0] = x[x>=0] - 1/2
    return y

def f1(x):
    x_all = np.array(x)
    y = np.empty_like(x)
    x = x_all[x_all<0]
    y[x_all<0] = -0.5*x*(x + 1)
    x = x_all[x_all>=0]
    y[x_all>=0] = 0.5*x*(x - 1)
    return y

def f2(x):
    x_all = np.array(x)
    y = np.empty_like(x)
    x = x_all[x_all<0]
    y[x_all<0] = -1/6*x**3 - 1/4*x**2 + 1/24
    x = x_all[x_all>=0]
    y[x_all>=0] = 1/6*x**3 - 1/4*x**2 + 1/24
    return y

def f3(x):
    x_all = np.array(x)
    y = np.empty_like(x)
    x = x_all[x_all<0]
    y[x_all<0] = -1/24*x**4 - 1/12*x**3 + 1/24*x
    x = x_all[x_all>=0]
    y[x_all>=0] = 1/24*x**4 - 1/12*x**3 + 1/24*x
    return y

eps = 0.123
print(-f0(-eps), f0(1-eps))
print(-f1(-eps), f1(1-eps))
x = np.linspace(-1, 1, 1000)
fig, ax = plt.subplots()
ax.plot(x, f0(x))
ax.plot(x, f1(x))
ax.plot(x, f2(x))
ax.plot(x, f3(x))
plt.show()

eps = np.linspace(0, 1, 1000)
for f in [f0, f1, f2, f3]:
    fig, ax = plt.subplots()
    ax.plot(eps, -f(-eps))
    ax.plot(eps, f(1-eps))
    plt.show()
    print(np.allclose(-f(-eps), f(1-eps)))
    # eps = 0.123456
    # print(-f(-eps), f(1-eps))
    
