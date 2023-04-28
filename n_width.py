# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:50:01 2022

@author: florianma
test resampling
plot 1/2sqrt(n)
jump weite oder smoothness

"""
from scipy.optimize import curve_fit
from scipy.linalg import svd, orth, qr
import numpy as np
import matplotlib.pyplot as plt
cmap = plt.cm.plasma
plt.close("all")


def get_large_step_matrix(N):
    X = np.zeros((N*4, N*4))
    for i in range(N):
        i1, i2, i3, i4 = np.sort([i, 2*N-1-i, 2*N+i, 4*N-1-i])
        X[i, [i1, i2, i3, i4]] = 0.5
        X[i, :i1] = 1.0
        X[i, i2:i3] = -1.0
        X[i, i4:] = 1.0
        X[i, [i1, i4]] = 0.5
        X[i, [i2, i3]] = -0.5
        # print(i, 2*N-1-i, 2*N+i, 4*N-1-i)
    X[N:2*N, :] = X[:N, :][::-1, :]
    X[2*N:, :] = -X[:2*N, :][::-1, :]
    return X


def u0(x, t, epsilon):
    y = np.zeros_like(x)
    # y[y < 0] = 0
    y[x > t] = 1
    # y[x >= t] = 1
    return 1-y


def u1(x, t, epsilon):
    y = .5+1/(2*epsilon)*(x-t)
    y[y < 0] = 0
    y[y > 1] = 1
    return y


def u2(x, t, epsilon):
    y = np.zeros_like(x)
    intervall1 = x <= (t-epsilon)
    intervall2 = ((t-epsilon) <= x) & (x <= t)
    intervall3 = (t <= x) & (x <= (t+epsilon))
    intervall4 = (t+epsilon) <= x
    y[intervall1] = 0
    y[intervall2] = ((x[intervall2]-(t-epsilon))/epsilon)**2 / 2
    y[intervall3] = 1-((x[intervall3]-(t+epsilon))/epsilon)**2 / 2
    y[intervall4] = 1
    return y


def get_snapshot_matrix(x, mu, u):
    M, N = len(x), len(mu)
    X = np.zeros((M, N))  # snapshot matrix
    for j in range(N):  # iteration over mus
        X[:, j] = u(x, mu[j])
    return X


def plot_snapshots_lines(x, X):
    m, n = X.shape
    fig, ax = plt.subplots()
    for i in range(n):
        # if i % 10 == 0:
        ax.step(x, X[:, i], "o--", where="mid")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    plt.show()


def plot_snapshots_2D(X):
    fig, ax = plt.subplots()
    ax.imshow(X, interpolation="nearest", vmin=-1, vmax=1)
    ax.set_aspect("auto")
    ax.set_xlabel("snapshot #")
    ax.set_ylabel("node #")
    ax.set_aspect("equal")
    plt.title("snapshot matrix")
    plt.show()


def plot_singular_values(X):
    U, S, VT = svd(X, full_matrices=False)
    # print(S)
    S = S/S[0]  # normalize singular values to make for a better comparison!

    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(0, len(S)), S, "ko-")
    ax1.set_xlabel("order")
    ax1.set_ylabel("singular value")
    ax1.set_yscale('log')
    ax1.set_xlim([0, len(S)])
    ax1.set_ylim([1e-16, 1e0])
    ax1.grid(which="both")
    plt.tight_layout()
    return S


def plot_basis_functions(x, U):
    fig, ax1 = plt.subplots()
    for i in range(len(U[0])):
        BF = U[:, i]
        ax1.plot(x, BF, "k.-", ms=.1, label="sv")
    ax1.set_xlabel("n")
    ax1.set_ylabel("Basis functions")
    ax1.set_xlim([0, 1])
    ax1.grid(which="both")
    plt.tight_layout()
    return


def sin_basis(x, N):
    M = len(x)
    dx = x[1] - x[0]
    U = np.zeros((M, N))
    R = N if N < M else M
    for i in range(R):
        A, w, p = guess(i, dx)
        p = 0
        U[:, i] = A * np.sin(w*x + p)
    return U


def constantins_basis_paper(x, N):
    M = len(x)
    U = np.zeros((M, N))
    for i in range(N):
        is_one = (i/N <= x) & (x < (i+1)/N)
        U[is_one, i] = 1.0
    assert np.all(np.sum(U, axis=1) == 1), "not orthogonal"
    return normalize(U)


def local_neighbours_average(x, u, r):
    mu = np.linspace(0, 1, r+1)
    print(r, mu[[0, -1]])
    U = get_snapshot_matrix(x, mu, u)
    W = (U[:, :-1]+U[:, 1:])/2
    Q, R = qr(W, mode="economic")
    return Q


def next_basis_vec(X, U):
    error = X - U @ (U.T @ X)
    L2 = np.mean(error**2, axis=0)**.5
    worst = np.argmax(L2)
    return normalize(error[:, worst]), L2[worst]


def greedy_basis(X, r=None):
    m, n = X.shape
    if not r:
        r = min(m, n)
    U = np.zeros_like(X[:, 0, None])
    err = np.zeros(r,)
    for i in range(r):
        psi_i, err[i] = next_basis_vec(X, U)
        U = np.c_[U, psi_i]
    return U
    # fig, ax = plt.subplots()
    # plt.plot(err, "b.")


def sinfunc(t, A, w, p, c=0):
    return A * np.sin(w*t + p) + c


def cosfunc(t, A, w, p, c=0):
    return A * np.cos(w*t + p) + c


def guess(i, dx):
    A = np.sin(np.pi/4)*2 * dx**.5
    w = (i/2+.25) * 2 * np.pi
    p = dx * 0.7853981633975*(1+2*i)
    return A, w, p


def guess2(i, dx, dm):
    A1 = np.sin(np.pi/4)*2 * dx**.5
    A2 = np.sin(np.pi/4)*2 * dm**.5
    w = (i/2+.25) * 2 * np.pi
    return A1, A2, w


def fit_sin(tt, yy, i, func):
    dx = tt[1] - tt[0]
    A_guess, w_guess, p_guess = guess(i, dx)
    popt, pcov = curve_fit(func, tt, yy, [A_guess, w_guess, p_guess])
    A, w, p = popt
    # print("correct guess? {:.6f} {:.6f} {:.6f} ".format(
    #     A-A_guess, w-w_guess, p-p_guess))
    return A, w, p


def singular_values(X):
    U, S, VT = svd(X, full_matrices=False)
    is_flipped = U[1, :] < 0
    U[:, is_flipped] *= -1
    VT[is_flipped, :] *= -1
    return U, S, VT


def fit(x, U, N, func):
    e1 = 0.0
    for i in reversed(range(N)):
        A, w, p = fit_sin(x, U[:, i], i, func)  # VT.T
        # print(i, A, w, p, sep="\t")
        # y = func(x, A, w, p)
        e1 += abs(p)
        # e2 += U[-1, i]-y[-1]
    return e1


def fit_and_plot(x, U, N, func):
    # dt = x[1]-x[0]
    # T = x.max() - x.min() + dt
    # print(T, "1.0?")
    fig, ax = plt.subplots()
    e = 0.0
    dx = x[1] - x[0]
    for i in reversed(range(N)):
        A, w, p = fit_sin(x, U[:, i], i, func)  # VT.T
        # print(i, A, w, p, sep="\t")
        y = func(x, A, w, p)
        e += np.mean((U[:, i]-y)**2)**.5
        ax.plot(x, U[:, i], "o", color=cmap(i/N))
        x_fine = np.linspace(x[0]-dx, x[-1]+dx, 10000)
        ax.plot(x_fine, func(x_fine, A, w, p), "--", color=cmap(i/N))
    plt.show()
    return e


def truncate(basis, r):
    return basis[:, :r]


def rms(X, basis):
    difference = X - (basis @ basis.T) @ X
    L2 = np.mean(difference**2, axis=0)**.5
    return L2


def calc_error(rb, X_test):
    M, N = X_test.shape
    R = N if N < M else M
    rms_error = np.ones((R,))
    max_error = np.ones((R,))
    for r in range(1, R, R//100):
        basis = rb(r)
        # fig, ax = plt.subplots(); plt.imshow(basis @ basis.T)
        L2 = rms(X_test, basis)
        rms_error[r] = np.sqrt(np.mean(L2**2))
        max_error[r] = L2.max()
        if r == 50:
            fig, ax = plt.subplots()
            ax.imshow(basis, interpolation="nearest")
            # print(basis)
            print(X_test)
            print(basis)
            print(L2)
    fig, ax = plt.subplots()
    ax.plot(max_error, ".", label="max_error")
    ax.plot(rms_error, ".", label="rms_error")
    ax.set_yscale('log')
    plt.ylim(1/100, 1)
    plt.legend()
    # plt.show()
    return rms_error, max_error

    # fig, ax = plt.subplots()
    # for r in [10, 100, 1000]:
    #     difference = X_fine - U_guess[:, :r] @ (U_guess[:, :r].T @ X_fine)
    #     L2 = np.sum(difference**2, axis=0)**.5 / len(x)
    #     fig, ax = plt.subplots()
    #     plt.hist(L2, 100)
    #     plt.title("L2 error distribution of random snapshots\n"
    #               "reduced rank = {:.0f}".format(r))
    #     plt.show()
    #     e1 = np.abs(difference).max()
    #     e2 = 1/X.size * np.sum(difference**2)**.5
    #     e3 = 1/X.size * np.sum(S[r:]**2)**.5
    #     print(r, e1, e2, e3)
    #     ax.plot(r, e1, "r.")
    #     ax.plot(r, e2, "g.")
    #     ax.plot(r, e3, "b.")
    # ax.set_yscale('log')
    # plt.show()


def extend_to_2pi(x, mu, X, U, S, VT):
    # x = np.linspace(dx/2, 4-dx/2, 4*M)
    # mu = np.linspace(dx/2, 4-dx/2, 4*M)
    M, N = X.shape
    M2, N2 = M*4, N*4
    dx2, dm2 = 4/M2, 4/N2
    x2 = np.linspace(dx2/2, 4-dx2/2, M*4)
    dx = x[1]-x[0]
    mu2 = np.linspace(dm2/2, 4-dm2/2, N*4)
    dm = mu[1]-mu[0]
    U2 = np.zeros((M*4, N))
    VT2 = np.zeros((N, N*4))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    for i in range(N):
        A1 = np.sin(np.pi/4)*2 * dx**.5
        A2 = np.sin(np.pi/4)*2 * dm**.5
        w1 = w2 = (i/2+.25) * 2 * np.pi
        [A, w, p], pcov = curve_fit(sinfunc, x, U[:, i], [A1, w1, 0])
        # A, w, p = A1, w1, 0
        print(i, A-A1, w-w1, p-0, sep="\t")
        U2[:, i] = A * np.sin(w*x2 + p)
        [A, w, p], pcov = curve_fit(cosfunc, mu, VT.T[:, i], [A2, w2, 0])
        # A, w, p = A2, w2, 0
        print(i, A-A2, w-w2, p-0, sep="\t")
        VT2.T[:, i] = A * np.cos(w*mu2 + p)

        # # A, w, p = guess(i, dx)
        # A1, A2, w = guess2(i, dx, dm)
        # A, w, p = fit_sin(x, U[:, i], i, sinfunc)
        # print(i, A, w, p, sep="\t")

        # # A, w, p = guess(i, dm)
        # A, w, p = fit_sin(mu, VT.T[:, i], i, cosfunc)
        # print(i, A, w, p, sep="\t")
        if i < 10:
            ax1.plot(x, U[:, i], "o", color=cmap(i/10))
            ax1.plot(x2, U2[:, i], "--", color=cmap(i/10))
            ax2.plot(mu, VT.T[:, i], "o", color=cmap(i/10))
            ax2.plot(mu2, VT2.T[:, i], "--", color=cmap(i/10))
    ax1.set_title("left singular values (U[:, i]), fit over extended domain")
    ax2.set_title("right singular values (VT[i, :]), fit over extended domain")
    ax1.set_xlim(0, 4)
    plt.show()
    X2 = U2 * S @ VT2
    plot_singular_values(X)
    plot_snapshots_2D(X)
    plot_snapshots_2D(X2)

    # fig, ax = plt.subplots()
    # ax.plot(x2, X2[:, 10])
    # ax.plot(x2, X2[:, 1010])
    # ax.plot(x2, X2[:, 2010])
    # ax.plot(x2, X2[:, 500])
    return X2, U2, S, VT2


def normalize(U):
    U_n = U / np.sum(U**2, axis=0)**.5
    return U_n


def orth2(W):
    W = W
    m, n = W.shape
    V = np.zeros_like(W)
    V[:, 0] = normalize(W[:, 0])
    for j in range(1, n):
        vi_wj = (V[:, :j] * W[:, j, None]).sum(axis=0)
        V[:, j] = normalize(W[:, j] - (vi_wj * V[:, :j]).sum(axis=1))
    return V


def sigmoid(x, time, a):
    return 1.0 / (1+np.e**((x-time)*a))


# if __name__ == "__main__":
#     M = 10  # samples
#     T = 1  # domain size
#     epsilon = 0.025  # jump size / 2

#     # mu = np.linspace(0, 1, M, endpoint=False)  # dolfin: 1 ... 0
#     # x = np.linspace(0, 1, M+1, endpoint=True)  # dolfin: 1 ... 0
#     dx = T/M  # = x[1] - x[0]
#     for s1 in [-dx, -dx/2, 0, dx/2, dx]:
#         for e1 in [1-dx, 1-dx/2, 1, 1+dx/2, 1+dx]:
#             for s2 in [-dx, -dx/2, 0, dx/2, dx]:
#                 for e2 in [1-dx, 1-dx/2, 1, 1+dx/2, 1+dx]:
#                     e1, e2 = f(s1, e1,  s2, e2)
#                     print(s1, e1, e1, e2, sep="\t")
#                     asd
#             # e1 = fit_and_plot(x, U, 10, sinfunc)
#             # e2 = fit_and_plot(mu, VT.T, 10, cosfunc)

#             # intervall01 = (0 <= x) & (x <= 1)
#             # fit_and_plot(x[intervall01], U[intervall01, :], 10, sinfunc)
#             # fit_and_plot(x[intervall01], VT.T[intervall01, :], 10, cosfunc)
