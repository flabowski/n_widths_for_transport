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
        # print(kwargs)
        test_sample = x.size//2
        mu = x[test_sample]
        print("mu_test = ", mu)
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


def move_x_intercept(u):
    def _impl(self, x, mu):
        x_ = x-mu+self.eps/2
        # we need to call u directly, since self.u is wrapped
        y = u(self, x_)
        y[x_ > self.eps] = 1.0
        y[x_ < 0] = 0.0
        return y
    return _impl


class CkRamp(Function):
    name = "C^k smooth ramp"

    def __init__(self, epsilon, k):
        self.eps = epsilon
        if k == 0:
            self.name = "smooth ramp, C^0"
            self.u = self.u0
        elif k == 1:
            self.name = "smooth ramp, C^1"
            self.u = self.u1
        elif k == 2:
            self.name = "smooth ramp, C^2"
            self.u = self.u2
        elif k == 3:
            self.name = "smooth ramp, C^3"
            self.u = self.u3
        elif k == 4:
            self.name = "smooth ramp, C^4"
            self.u = self.u4
        elif k == 5:
            self.name = "smooth ramp, C^5"
            self.u = self.u5
        else:
            raise NotImplementedError
        return

    @move_x_intercept
    def u0(self, x):
        y = 1/self.eps * x
        return y

    @move_x_intercept
    def u1(self, x):
        e = self.eps
        y = -2/e**3 * x**3 + 3/e**2*x**2
        return y

    @move_x_intercept
    def u2(self, x):
        e = self.eps
        y = 6/e**5*x**5 - 15/e**4*x**4 + 10/e**3*x**3
        return y

    @move_x_intercept
    def u3(self, x):
        e = self.eps
        y = -20/e**7 * x**7 + 70/e**6 * x**6 - 84/e**5 * x**5 + 35/e**4 * x**4
        return y

    @move_x_intercept
    def u4(self, x):
        e = self.eps
        y = (70/e**9*x**9 - 315/e**8*x**8 + 540/e**7*x**7
             - 420/e**6*x**6 + 126/e**5*x**5)
        return y

    @move_x_intercept
    def u5(self, x):
        e = self.eps
        y = (-252/e**11*x**11 + 1386/e**10*x**10 - 3080/e**9*x**9
             + 3465/e**8*x**8 - 1980/e**7*x**7 + 462/e**6*x**6)
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
    x = Domain([0, 1], 1000)
    mu = Domain([0, 1], 100)
    u0 = CkRamp(0.025*2, 0)
    u1 = CkRamp(0.025*2, 1)
    u2 = CkRamp(0.025*2, 2)
    u3 = CkRamp(0.025*2, 3)
    u4 = CkRamp(0.025*2, 4)

    u5 = Heaviside()
    u6 = LinearRamp(0.025)
    u7 = SmoothRamp(0.025)
    u8 = Sigmoid(50)

    y_old = u7(x(), 0.5)
    y_new = u1(x(), 0.5)
    fig, ax = plt.subplots()
    ax.plot(x(), y_old, "go", label="alt (2 intervalle)")
    ax.plot(x(), y_new, "r.", label="neu (p3)")
    ax.legend()
    plt.show()
    for u in [u1, u2, u3, u4, u5]:
        print(u.name)
        u.plot(x())
        X = u(x(), mu())
        fig, ax = plt.subplots()
        ax.imshow(X, interpolation="nearest")
        plt.show()
