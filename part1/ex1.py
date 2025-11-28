import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return np.square(x) + np.square(y-1) + np.power(x-y, 4)

def himmelblau(x, y):
    return np.square(np.square(x)+y-11) + np.square(x+np.square(y))

def rosenbrock(x, y):
    a = 1
    b = 100
    return np.square(a-x) + b*np.square(y-np.square(x))

def GD():
    pass

def Newton():
    pass

X = np.linspace(-100, 100, 1000, dtype=np.float32)
Y = np.linspace(-100, 100, 1000, dtype=np.float32)

x, y = np.meshgrid(X, Y)

