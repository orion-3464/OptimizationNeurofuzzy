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
    x_prev = np.ones(2)*1000
    iterations = 0

    for i in range(max_iter):
        iterations = iterations + 1 
        x = x - lr*grad(x[0], x[1], A)
        #print(f"Iteration {i+1}: x = {x[0]:.4f}, y = {x[1]:.4f}")
        
        if np.linalg.norm(x-x_prev, ord=2) < 1e-4:
            print(f"Converged (took {i+1} iterations)")
            break
        print(grad(x[0], x[1], A))
        x_prev = x.copy()
        path.append(x)
        
    return x, path, iterations

def Newton(grad, hessian, A, x=np.zeros(2, dtype=np.float32), max_iter=100):
    x = x.copy()
    path = [x]
    x_prev = np.ones(2)*1000
    iterations = 0

    for i in range(max_iter):
        g = grad(x[0], x[1], A)
        H = hessian(A)
        iterations = iterations + 1
        
        try:
            delta = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered, stopping.")
            break
        
        x = x + delta      
        #print(f"Iteration {i+1}: x = {x[0]:.4f}, y = {x[1]:.4f}")
        
        if np.linalg.norm(x-x_prev, ord=2) < 1e-4:
            print(f"Converged (took {i+1} iterations)")
            break
        
        x_prev = x.copy()
        path.append(x)
        
    return x, path, iterations

if __name__ == "__main__":
    A = np.linspace(1, 101, 1000)
    X = np.linspace(-2, 2, 1000, dtype=np.float32)
    Y = np.linspace(-2, 2, 1000, dtype=np.float32)

    np.random.seed(17)
    x0 = np.random.choice(X)
    y0 = np.random.choice(Y)
    start = np.array([x0, y0])

    _, _, iters_gd = GD(f_grad, 5000, x=start)
    _, _, iters_nt = Newton(f_grad, f_hessian, 1000000, x=start)

