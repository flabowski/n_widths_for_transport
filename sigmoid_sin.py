# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 09:59:48 2023

@author: florianma
"""
import sympy as sp
x = sp.Symbol('x')
i = sp.Symbol('i', integer=True)
eps = sp.Symbol('eps')
pi = sp.pi
sin = sp.sin
m = pi**4/2**4
f1 = sin(pi/2*x)  # /m*2/eps
f2 = sin(pi/2*sin(pi/2*x))
f3 = sin(pi/2*sin(pi/2*sin(pi/2*x)))
f4 = sin(pi/2*sin(pi/2*sin(pi/2*sin(pi/2*x))))
df1 = sp.simplify(sp.Derivative(f1, x))
df2 = sp.simplify(sp.Derivative(f2, x))
df3 = sp.simplify(sp.Derivative(f3, x))
df4 = sp.simplify(sp.Derivative(f4, x))
print(df1)
print(df2)
print(df3)
print(df4)
print(df4.subs(x, 0))

import matplotlib.pyplot as plt
import numpy as np
plt.close("all")

def u(x, mu, k, eps):
    # eps = 0.025
    sin, pi = np.sin, np.pi
    m0 = pi**k/2**k
    x_ = (x-mu)/m0*2/eps
    for i in range(k):
        y = sin(pi/2 * x_)
        x_ = y
    # y[x < mu-m*eps/2] = -1
    # y[x > mu+m*eps/2] = 1
    # m_0 = 
    m_a = (np.pi**k/2**k) * 1/m0*2/eps
    T = 2*np.pi / (np.pi/2) / (1/m0) / (2/eps)
    T = 2*m0*eps
    print(k, m_a, T)
    return y/2+0.5


x = np.linspace(-np.pi, np.pi, 1000)
x = np.linspace(-1, 1, 1000)
mu = 0
eps = .025
k = 4
m0 = np.pi**k / 2**k
x_ = x * 1/m0*2/eps
for k in range(1, 5):
    y = np.sin(np.pi/2 * x_)
    plt.plot(x, y)
    x_ = y
    m_a = (np.pi**k/2**k) * 1/m0*2/eps
    T = 2*np.pi / (np.pi/2) / (1/m0) / (2/eps)
    T = 2*m0*eps
    dy = y[500]-y[499]
    dx = x[500]-x[499]
    m_n = dy/dx
    print(k, m_n, m_a, T)
    # plt.plot(x, np.sin(np.pi/2 * np.sin(np.pi/2*x)))
    # plt.plot(x, np.sin(np.pi/2*np.sin(np.pi/2*np.sin(np.pi/2*x))))

eps=0.05
k = np.arange(10)
m = np.pi**k/2**k
T = 2*m*eps
print(k)
print(m)
print(T)

fig, ax = plt.subplots()
x = np.linspace(-1, 1, 1000)
for k in [2, 3, 4, 5, 6]:
    mu = 0
    y = u(x, mu, k, eps=0.025)
    plt.plot(x, y)
plt.show()
# x -> 2 pi
# pi/2 x -> 4
# pi/2*x/m*2/eps -> m*eps/2

# plt.plot(x, np.sin(x))  # 2 pi
# plt.plot(x, np.sin(2*np.pi*x))  # 1
# plt.plot(x, np.sin(np.pi*x))  # 2
# plt.plot(x, np.sin(np.pi/2*x))  # 4
