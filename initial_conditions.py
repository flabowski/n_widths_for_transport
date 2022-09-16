# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:49:18 2022

@author: florianma
"""
import numpy as np
import matplotlib.pyplot as plt


class Domain:
    def __init__(self, bounds, size):
        self.bounds = bounds
        self.size = size
        self.min, self.max = bounds[0], bounds[1]
        T = self.max - self.min
        self.delta_x = dx = T/size
        self.x = np.linspace(self.min+dx/2, self.max-dx/2, size)
        return

    def __call__(self):
        return self.x


class Function:
    name = None

    def __init__(self):
        return

    def __call__(self, x, mu):
        if isinstance(mu, float):
            return self.u(x, mu)
        m, n = x.size, mu.size
        X = np.zeros((m, n))  # snapshot matrix
        for j, mu_j in enumerate(mu):
            X[:, j] = self.u(x, mu_j)
        return X

    def u(self, x, mu):
        return

    def plot(self, x, ax=None, **kwargs):
        print(kwargs)
        test_sample = x.size//2
        mu = x[test_sample]
        print(mu)
        if not ax:
            fig, ax = plt.subplots()
        ax.plot(x, self.u(x, mu), ".", **kwargs)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x; mu={:.4f})".format(mu))
        if not ax:
            plt.show()
        return


class Heaviside(Function):
    name = "heaviside"

    def u(self, x, mu):
        y = np.zeros_like(x)
        y[x < mu] = 0.0
        y[x == mu] = .5
        y[x > mu] = 1.0
        return y


class LinearRamp(Function):
    name = "linear ramp"

    def __init__(self, epsilon):
        self.eps = epsilon
        return

    def u(self, x, mu):
        eps = self.eps
        y = .5+1/(2*eps)*(x-mu)
        y[y < 0] = 0
        y[y > 1] = 1
        return y


class SmoothRamp(Function):
    name = "smooth ramp"

    def __init__(self, epsilon):
        self.eps = epsilon
        return

    def u(self, x, mu):
        epsilon = self.eps
        y = np.zeros_like(x)
        intervall1 = x <= (mu-epsilon)
        intervall2 = ((mu-epsilon) <= x) & (x <= mu)
        intervall3 = (mu <= x) & (x <= (mu+epsilon))
        intervall4 = (mu+epsilon) <= x
        y[intervall1] = 0
        y[intervall2] = ((x[intervall2]-(mu-epsilon))/epsilon)**2 / 2
        y[intervall3] = 1-((x[intervall3]-(mu+epsilon))/epsilon)**2 / 2
        y[intervall4] = 1
        return y


class Sigmoid(Function):
    name = "sigmoid"

    def __init__(self, a):
        self.a = a
        return

    def u(self, x, mu):
        a = self.a
        return 1.0 / (1+np.e**(-(x-mu)*a))


if __name__ == "__main__":
    plt.close("all")
    x = Domain([0, 1], 100)
    mu = Domain([0, 1], 100)
    u1 = Heaviside()
    u2 = Ramp(0.025)
    u3 = SmoothJump(0.025)
    u4 = Sigmoid(50)
    for u in [u1, u2, u3, u4]:
        print(u.name)
        u.plot(x())
        X = u(x(), mu())
        fig, ax = plt.subplots()
        ax.imshow(X, interpolation="nearest")
        plt.show()
