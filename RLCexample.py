import numpy as np
import matplotlib.pyplot as plt
import os

def f(t, y):
  R = 20
  L = 1e-3
  C = 2.53e-9
  ss = np.array([[-R/L, -1/(L*C)], [1, 0]])
  return np.matmul(ss, y)

figFolder = 'RLCfig'

# Euler Method

h = 10e-9
t = np.arange(0, 500e-6, h)
n = len(t)

y = np.zeros([n, 2])
y[0] = [5, 0]

for i in range(n-1):
  y[i + 1] = y[i] + f(t[i], y[i]) * h

plt.plot(t, np.transpose(y)[0])
plt.savefig(os.path.join(figFolder, 'euler.png'))
plt.show()

# Runge-Kutta Method

h = 100e-9
t = np.arange(0, 500e-6, h)
n = len(t)

y = np.zeros([n, 2])
y[0] = [5, 0]

for i in range(n-1):
  k_1 = f(t[i], y[i])
  k_2 = f(t[i] + (h/2), (y[i] + k_1*(h/2)))
  k_3 = f(t[i] + (h/2), (y[i] + k_2*(h/2)))
  k_4 = f(t[i] + h, (y[i] + k_3*h))
  k = (k_1 + 2*k_2 + 2*k_3 + k_4) / 6
  y[i + 1] = y[i] + k*h

plt.plot(t, np.transpose(y)[0])
plt.savefig(os.path.join(figFolder, 'rk.png'))
plt.show()

