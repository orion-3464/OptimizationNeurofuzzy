import numpy as np
import matplotlib.pyplot as plt

def f(x, y, A):
    return A*x**2 + (1/A)*y**2

def f_grad(x, y, A):
    gradx = 2*A*x
    grady = (2/A)*y
    return np.array([gradx, grady], dtype=np.float32)

def f_hessian(A): # Hessian independent of x, y
    return np.array([[2*A, 0], 
                     [0, 2/A]], dtype=np.float32)

def GD(grad, A, x=np.zeros(2, dtype=np.float32), lr=1e-3, max_iter=10000):
    x = x.copy()
    path = [x]
    H = f_hessian(A)

    for i in range(max_iter):
        g = grad(x[0], x[1], A)
        if np.linalg.norm(g) < 1e-6:
            break
        
        denom = np.dot(g, np.dot(H, g))
        lr = np.dot(g, g) / denom if denom != 0 else 0

        x = x - lr * g
        path.append(x)
        
    return i

def visualize(A_values, start_point):    
    condition_numbers = []
    iteration_counts = []

    x = np.linspace(-2, 2, 500)
    y = np.linspace(-2, 2, 500)
    X, Y = np.meshgrid(x, y)

    for A in A_values:
        iters = GD(f_grad, A, start_point)
        H = f_hessian(A)
        kappa = np.linalg.cond(H)
        
        condition_numbers.append(kappa)
        iteration_counts.append(iters)

    plt.plot(condition_numbers, iteration_counts, 'r-s', lw=2, markersize=8)
    plt.title("Number of Iterations to Converge")
    plt.xlabel("Condition Number (κ)")
    plt.ylabel("Iterations")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    A = np.linspace(2, 10, 9)
    x0 = np.array([-1.8, 1.8], dtype=np.float32)
   
    visualize(A, x0)