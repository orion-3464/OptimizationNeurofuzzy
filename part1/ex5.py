import numpy as np
import matplotlib.pyplot as plt

A_POINTS = np.array([1.0, -1.0, -1.0, 1.0])
B_POINTS = np.array([1.0, 1.0, -1.0, -1.0])

def f(x1, x2):
    total = 0
    for i in range(4):
        total += (x1 - A_POINTS[i])**2 + (x2 - B_POINTS[i])**2
    return 0.25 * total

def get_stochastic_grad(x1, x2, i):
    return 0.5 * np.array([x1 - A_POINTS[i], x2 - B_POINTS[i]])

def SGD(x_start, lr=0.1, max_iter=200, schedule=False):
    x = x_start.copy()
    path = [x.copy()]
    
    for k in range(1, max_iter + 1):
        current_lr = lr / np.sqrt(k) if schedule else lr
        
        i = np.random.randint(0, 4)
        
        grad = get_stochastic_grad(x[0], x[1], i)
        x = x - current_lr * grad
        
        path.append(x.copy())
        
    return x, path

def Visualise_SGD(X, Y, path, title):
    xg, yg = np.meshgrid(X, Y)
    Z = f(xg, yg)
    path = np.array(path)

    plt.figure(figsize=(8, 6))
    cnt = plt.contour(xg, yg, Z, levels=20)
    plt.colorbar(cnt)
    
    plt.plot(path[:,0], path[:,1], 'b.-', alpha=0.6, label='SGD Path')
    plt.scatter(path[0,0], path[0,1], c='green', label='Start', s=100)
    plt.scatter(path[-1,0], path[-1,1], c='red', label='End', s=100)
    
    plt.scatter(A_POINTS, B_POINTS, c='black', marker='x', label='Points (a_i, b_i)')

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    X_range = np.linspace(-2, 2, 100)
    Y_range = np.linspace(-2, 2, 100)
    start_point = np.array([1.5, 1.5])

    x_f1, path1 = SGD(start_point, lr=0.05)
    Visualise_SGD(X_range, Y_range, path1, "SGD with lr = 0.05")

    x_f2, path2 = SGD(start_point, lr=0.2)
    Visualise_SGD(X_range, Y_range, path2, "SGD with lr = 0.2")

    x_f3, path3 = SGD(start_point, lr=0.2, schedule=True)
    Visualise_SGD(X_range, Y_range, path3, "SGD: With lr Scheduling Starting with lr = 0.2")