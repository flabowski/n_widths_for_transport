# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 09:19:21 2022

@author: florianma
"""
import time
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


b = 3  # #snapshots
r = 100
x = sp.Symbol('x')
mu = sp.Symbol('mu')
f = sp.Heaviside(x-mu)
f = 1/(1+sp.exp(-x))

n = np.arange(r)
i = n[1:]
the_sum = np.cumsum(1/(2*i - 1)**2)
estimate2 = (1/2 - 4/np.pi**2 * the_sum) ** 0.5

A = 2**.5  # sp.sqrt(2) is slow
T = 1
delta_n = np.zeros(r,)
term1 = 0.5  # sp.integrate(sp.integrate(f*f, (x, 0, 1)), (mu, 0, 1))
term2 = 0
for i in range(r):  # rank
    # print("i = ", i)
    omega = 2 * sp.pi/(4*T) * (2*i+1)
    psi_r = A * sp.sin(omega*x)
    t1 = time.time()
    res = sp.integrate(f*psi_r, (x, 0, 1))
    t2 = time.time()
    res2 = sp.integrate(res**2, (mu, 0, 1))
    t3 = time.time()
    res3 = res2.evalf()
    t4 = time.time()
    term2 += res3
    delta_n[i] = term1-term2
    # print(res)
    # print(res**2)
    # print((res**2).evalf())
    # print(res2)
    print(i, res3, term2)
    # print(t2-t1, t3-t2, t4-t3)

lbl2 =  r'$\sqrt {\left( \frac{1}{2} - \frac{4}{\pi^2} \sum_{i=1}^N  \frac{1}{(2i-1)^2 } \right)}$'


fig, ax = plt.subplots()
ax.plot(n, delta_n**.5, "b.", ms=2, label="POD basis")
# ax.plot(estimate1, "k--", label=lbl1, lw=.5)
ax.plot(estimate2, "r--", label=lbl2, lw=.5)
plt.legend(prop={'size': 8})
ax.set_yscale('log')
plt.ylim([1e-2, 1])
plt.xlim([0, 500])
plt.xlabel("order")
plt.ylabel("rmse")
plt.grid(which="both")
plt.show()


asd
approximation = 0
# X_reduced = r*[None]
mus = np.random.rand(b,)
C = np.empty((r, b))
X = b*[0]
X_ = b*[0]
for j in range(b):
    X[j] = sp.Heaviside(x-mus[j])
for i in range(r):  # rank
    print("i = ", i)
    omega = 2 * sp.pi/(4*T) * (2*i+1)
    u_j = A * sp.sin(omega*x)
    c = sp.integrate(u_j*f, (x, 0, 1))  # depends on mu
    for j in range(b):
        t1 = time.time()
        C[i, j] = c.subs(mu, mus[j]).evalf()
        X_[j] += u_j*C[i, j]

        if i in [5, 10]:
            error = X[j]-X_[j]
            t2 = time.time()
            print("err", t2-t1)
            # term1 = sp.integrate(X[j]**2, (x, 0, 1))**0.5
            term1 = 1.0
            t3 = time.time()
            # print("integrate t1", t3-t2, term1)
            term2 = sp.integrate((2*X[j]*X_[j]).evalf(), (x, 0, 1)).evalf()**0.5
            t4 = time.time()
            print("integrate t2", t4-t3, term2)
            term3 = sp.integrate(X_[j]**2, (x, 0, 1))**0.5
            t5 = time.time()
            print("integrate t3", t5-t4, term3)
            # norm = sp.integrate(error**2, (x, 0, 1))**0.5
            norm = (term1+term2+term3).evalf()
            t6 = time.time()
            print("integrate error", t6-t5)
            # print((term1+term2+term3).evalf(), norm.evalf())



    #     d_n = sp.integrate(norm, (mu, 0, 1))
    #     print(d_n)
    #     # print(X_[j])
    # print(c)
    # print(c.subs(mu, 0.123456).evalf())
    # # X_reduced[i] = c
    # approximation += u_j*c  # outer product
    # # sp.plot(approximation.subs(mu, 0.45), (x, 0, 1))
    # if i == 5:
    #     error = f-approximation
    #     print(error)
    #     norm = sp.integrate(error**2, (x, 0, 1))**0.5
    #     print(norm)
    #     d_n = sp.integrate(norm, (mu, 0, 1))
    #     print(d_n)
    # # print("d_n =", d_n)
    # # sp.plot(X_reduced, (mu, 0, 1))

asd

# discrete
import numpy as np
import matplotlib.pyplot as plt
a, b, r = 1000, 30, 10
r = max(a, b, r)
dx, dm = 1/a, 1/b
x = np.linspace(dx/2, 1-dx/2, a)
mu = np.linspace(dm/2, 1-dm/2, b)


def u(x, mu):
    y = np.zeros_like(x)
    y[x < mu] = 0.0
    y[x == mu] = .5
    y[x > mu] = 1.0
    return y


X = np.zeros((a, b))
for j in range(b):
    X[:, j] = u(x, mu[j])

fig, ax = plt.subplots()

X_reduced = np.zeros((r, b))
U_, S, VT = np.linalg.svd(X, full_matrices=False)
U = np.zeros((a, r))
A = (2*dx)**.5
# A = 2*dx**0.5 * np.sin(np.pi/4)
approximation = 0
for j in range(r):
    omega = 2 * np.pi/(4*T) * (2*j+1)
    u_j = A * np.sin(omega*x)  # a, (depends on x)
    print(j, np.sum(u_j**2))
    U[:, j] = u_j
    c = np.sum(u_j[:, None]*X, axis=0)  # b, (depends on mu)
    X_reduced[j, :] = c
    approximation += u_j[:, None]*c[None, :]

    appr = U @ (U.T@X)
    ax.plot(x, appr[:, 13], "k-")  # mu = 0.45
    ax.plot(x, approximation[:, 13], ".--")  # mu = 0.45

plt.show()
# plt.imshow(U@U.T)   # approximation
# plt.imshow(U.T@U)  # unit matrix
# plt.imshow(U@U.T)
for i in range(a):
    x_i = x[i]
    for j in range(b):
        mu_j = mu[j]
        x_ij = u(x_i, mu_j)
        for k in range(r):
            omega = 2 * np.pi/(4*T) * (2*k+1)
            u_k = A * np.sin(omega*x)
            # res[i, j, k] = 
