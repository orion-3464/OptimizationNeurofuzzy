import numpy as np
import matplotlib.pyplot as plt

def f(x1, x2):
    return np.max(np.array([
        x1 + x2, 
        0.9*x1 - 1.1*x2 + 1,
        -0.8*x1 + 1.2*x2 - 1,
        2 - 1.1*x1 - 0.9*x2
    ]), axis=0)

def grad_f(x1, x2):
    vals = np.array([
        x1 + x2, 
        0.9*x1 - 1.1*x2 + 1,
        -0.8*x1 + 1.2*x2 - 1,
        2 - 1.1*x1 - 0.9*x2
    ])
    idx = np.argmax(vals)
    pot_gradients = np.array([[1.0, 1.0], [0.9, -1.1], [-0.8, 1.2], [-1.1, -0.9]], dtype=np.float32)
    return pot_gradients[idx]

def GD(grad, x, lr=0.1, max_iter=2000, schedule='fixed'):
    x = x.astype(np.float32).copy()
    path = [x.copy()]
    for i in range(1, max_iter + 1):
        step = lr / i if schedule == '1/k' else lr
        g = grad(x[0], x[1])
        x_new = x - step * g
        if np.linalg.norm(x_new - x) < 1e-7:
            break
        x = x_new
        path.append(x.copy())
    return x, path

def Visualise(func, X, Y, path, limits, title):
    xg, yg = np.meshgrid(X, Y)
    Z = func(xg, yg)
    path = np.array(path)
    plt.figure(figsize=(8, 6))
    plt.contourf(xg, yg, Z, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar()
    plt.contour(xg, yg, Z, levels=20, colors='black', linewidths=0.5)
    if len(path) > 0:
        plt.plot(path[:,0], path[:,1], 'r-', linewidth=1)
        plt.scatter(path[0,0], path[0,1], c='green', s=50)
        plt.scatter(path[-1,0], path[-1,1], c='white', edgecolor='black', s=50)
    plt.title(title)
    plt.xlim(limits[0], limits[1])
    plt.ylim(limits[2], limits[3])
    plt.show()

if __name__ == "__main__":
    X = np.linspace(-5, 5, 200)
    Y = np.linspace(-5, 5, 200)
    start = np.array([-4.0, 4.0])

    x_fix, path_fix = GD(grad_f, x=start, lr=0.5, schedule='fixed')
    Visualise(f, X, Y, path_fix, [-5, 5, -5, 5], "Fixed Step (lr=0.5)")

    x_fix, path_fix = GD(grad_f, x=start, lr=0.1, schedule='fixed')
    Visualise(f, X, Y, path_fix, [-5, 5, -5, 5], "Fixed Step (lr=0.1)")

    x_dec, path_dec = GD(grad_f, x=start, lr=1.0, schedule='1/k')
    Visualise(f, X, Y, path_dec, [-5, 5, -5, 5], "Step 1/k")
