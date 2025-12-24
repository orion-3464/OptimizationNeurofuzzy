import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 1000
A_POINTS = np.random.uniform(-2, 2, N)
B_POINTS = np.random.uniform(-2, 2, N)

def f(x1, x2):
    total = 0
    for i in range(N):
        total += (x1 - A_POINTS[i])**2 + (x2 - B_POINTS[i])**2
    return (1/N) * total

def get_batch_grad(x1, x2, indices):
    grad_x1 = np.mean(x1 - A_POINTS[indices])
    grad_x2 = np.mean(x2 - B_POINTS[indices])
    return np.array([grad_x1, grad_x2])

def MiniBatchSGD(x_start, lr=0.1, batch_size=32, max_iter=200):
    x = x_start.copy()
    path = [x.copy()]
    
    for _ in range(max_iter):
        indices = np.random.choice(N, batch_size, replace=False)
        
        grad = get_batch_grad(x[0], x[1], indices)
        x = x - lr * grad
        
        path.append(x.copy())
    return x, path

if __name__ == "__main__":
    X_range = np.linspace(-2, 2, 100)
    Y_range = np.linspace(-2, 2, 100)
    start_point = np.array([1.8, 1.8])

    sizes = [1, 32, 256]
    for b in sizes:
        xf, path = MiniBatchSGD(start_point, lr=0.1, batch_size=b)
        
        xg, yg = np.meshgrid(X_range, Y_range)
        Z = f(xg, yg)
        plt.figure(figsize=(7, 5))
        plt.contour(xg, yg, Z, levels=20)
        path = np.array(path)
        plt.plot(path[:,0], path[:,1], '.-', label=f'Batch Size = {b}')
        plt.title(f"Mini-batch SGD (Size={b})")
        plt.legend()
        plt.show()