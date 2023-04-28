# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 10:13:39 2023

@author: florianma
"""
import numpy as np
import matplotlib.pyplot as plt
cmap = plt.cm.plasma

def plot_modes(x, X, r=10):
    for i in range(r):
        print(i, r, i/r)
        plt.plot(x, X[:, i], "o", ms=1, color=cmap(i/(r-1)), label="mode {:.0f}".format(i))
    plt.xlim([0, 1])
    plt.grid(which="both")
    return

def plot_and_compare_modes(x, U1, U2, r=6):
    fig, ax1 = plt.subplots()
    for i in range(r):
        plt.plot(x, U1[:, i], "o", ms=1, color=cmap(i/(r-1)), label="mode {:.0f}".format(i))
    for i in range(r):  # fix z-order
        plt.plot(x, U2[:, i], "k--")
    plt.xlim([0, 1])
    plt.grid(which="both")
    plt.show()
    return

def plot_singular_values(S):
    S = S/S[0].copy()  # normalize singular values to make for a better comparison!
    fig, ax1 = plt.subplots()
    ax1.plot(np.arange(1, len(S)+1), S, "k.", ms=1)
    ax1.set_xlabel("order")
    ax1.set_ylabel("singular value")
    ax1.set_yscale('log')
    ax1.set_xlim([0, len(S)])
#    ax1.set_ylim([1e-6, 100])
    ax1.grid(which="both")
    plt.tight_layout()
    plt.show()
    return fig, ax1

def plot_error(rms_error, max_error):
    fig, ax = plt.subplots()
    ax.plot(max_error, ".", label="max_error")
    ax.plot(rms_error, ".", label="rms_error")
    ax.set_yscale('log')
    plt.legend(prop={'size': 8})
    return fig, ax

def plot_paramspace(X):
    fig, ax = plt.subplots()
    plt.imshow(X, interpolation="none")
    ax.set_xlabel("$\mu$")
    ax.set_ylabel("$x$")
    plt.show()
    return fig, ax

def save_im(res, name):
    print(res.min(), res.max())
    fig, ax = plt.subplots()
    cs = plt.imshow(res, interpolation="nearest", origin="upper")
    plt.colorbar(cs)
    a = np.round(cmap(res/2+0.5)*256, decimals=0)
    img = np.array(a-1, dtype=np.uint8)
    from PIL import Image
    im = Image.fromarray(img)
    im.save(name+'.png')
    plt.close()