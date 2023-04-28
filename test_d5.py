# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 18:09:30 2023

@author: florianma
"""
import numpy as np
import matplotlib.pyplot as plt

e = 0.025
mu = 0.5
a = 100000
x0 = np.linspace(0, 1, a)
x = x0-mu

y = (-252/e**11*x**11 + 1386/e**10*x**10 - 3080/e**9*x**9
     + 3465/e**8*x**8 - 1980/e**7*x**7 + 462/e**6*x**6)


dy1_a = (- 252./e**11*x**10 * 11
      + 1386/e**10*x**9 * 10
      - 3080/e**9*x**8 * 9
      + 3465/e**8*x**7 * 8
      - 1980/e**7*x**6 * 7
      + 462./e**6*x**5 * 6)
dy2_a = (- 252./e**11*x**9 * 11*10
      + 1386/e**10*x**8 * 10*9
      - 3080/e**9*x**7 * 9*8
      + 3465/e**8*x**6 * 8*7
      - 1980/e**7*x**5 * 7*6
      + 462./e**6*x**4 * 6*5)
dy3_a = (- 252./e**11*x**8 * 11*10*9
      + 1386/e**10*x**7 * 10*9*8
      - 3080/e**9*x**6 * 9*8*7
      + 3465/e**8*x**5 * 8*7*6
      - 1980/e**7*x**4 * 7*6*5
      + 462./e**6*x**3 * 6*5*4)
dy4_a = (- 252./e**11*x**7 * 11*10*9*8
      + 1386/e**10*x**6 * 10*9*8*7
      - 3080/e**9*x**5 * 9*8*7*6
      + 3465/e**8*x**4 * 8*7*6*5
      - 1980/e**7*x**3 * 7*6*5*4
      + 462./e**6*x**2 * 6*5*4*3)
dy5_a = (- 252./e**11*x**6 * 11*10*9*8*7
      + 1386/e**10*x**5 * 10*9*8*7*6
      - 3080/e**9*x**4 * 9*8*7*6*5
      + 3465/e**8*x**3 * 8*7*6*5*4
      - 1980/e**7*x**2 * 7*6*5*4*3
      + 462./e**6*x**1 * 6*5*4*3*2)
dy6_a = (- 252./e**11*x**5 * 11*10*9*8*7*6
      + 1386/e**10*x**4 * 10*9*8*7*6*5
      - 3080/e**9*x**3 * 9*8*7*6*5*4
      + 3465/e**8*x**2 * 8*7*6*5*4*3
      - 1980/e**7*x**1 * 7*6*5*4*3*2
      + 462./e**6*x**0 * 6*5*4*3*2*1)

y[x > e] = 1.0
y[x < 0] = 0.0
dy1_a[x > e] = 0.0
dy1_a[x < 0] = 0.0
dy2_a[x > e] = 0.0
dy2_a[x < 0] = 0.0
dy3_a[x > e] = 0.0
dy3_a[x < 0] = 0.0
dy4_a[x > e] = 0.0
dy4_a[x < 0] = 0.0
dy5_a[x > e] = 0.0
dy5_a[x < 0] = 0.0
dy6_a[x > e] = 0.0
dy6_a[x < 0] = 0.0

dy1_n = np.diff(y, n=1) * a
dy2_n = np.diff(y, n=2) * a**2
dy3_n = np.diff(y, n=3) * a**3
dy4_n = np.diff(y, n=4) * a**4
dy5_n = np.diff(y, n=5) * a**5
dy6_n = np.diff(y, n=6) * a**6


x1 = (x[1:] + x[:-1])/2
x2 = (x1[1:] + x1[:-1])/2
x3 = (x2[1:] + x2[:-1])/2
x4 = (x3[1:] + x3[:-1])/2
x5 = (x4[1:] + x4[:-1])/2
x6 = (x5[1:] + x5[:-1])/2

fig, ax = plt.subplots()
plt.plot(x, dy1_a)
plt.plot(x1, dy1_n)
plt.xlim([-e, 2*e])
plt.show()

fig, ax = plt.subplots()
plt.plot(x, dy2_a)
plt.plot(x2, dy2_n)
plt.xlim([-e, 2*e])
plt.show()

fig, ax = plt.subplots()
plt.plot(x, dy3_a)
plt.plot(x3, dy3_n)
plt.xlim([-e, 2*e])
plt.show()

fig, ax = plt.subplots()
plt.plot(x, dy4_a)
plt.plot(x4, dy4_n)
plt.xlim([-e, 2*e])
plt.show()

fig, ax = plt.subplots()
plt.plot(x, dy5_a)
plt.plot(x5, dy5_n)
plt.xlim([-e, 2*e])
plt.show()

fig, ax = plt.subplots()
plt.plot(x, dy6_a)
plt.plot(x6, dy6_n)
plt.xlim([-e, 2*e])
plt.show()


print(np.mean(y**2, axis=0)**.5)
print(np.mean(dy1_n**2, axis=0)**.5)
print(np.mean(dy2_n**2, axis=0)**.5)
print(np.mean(dy3_n**2, axis=0)**.5)
print(np.mean(dy4_n**2, axis=0)**.5)
print(np.mean(dy5_n**2, axis=0)**.5)
print(np.mean(dy6_n**2, axis=0)**.5)