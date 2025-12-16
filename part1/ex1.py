import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return np.square(x) + np.square(y-1) + np.power(x-y, 4)

def grad_f(x, y):
    gradx = 2*x + 4*(x-y)**3
    grady = 2*(y-1) - 4*(x-y)**3
    return np.array([gradx, grady])

def grad2_f(x, y):
    return 4 + 24*(x-y)**2 + 1e-6 # avoid making zero denominator
    
def himmelblau(x, y):
    return np.square(np.square(x)+y-11) + np.square(x+np.square(y))

def himmelblau_grad(x, y):
    gradx = 4*(x**2+y-11)*x + 2*(x+y**2)
    grady = 2*(x**2+y-11) + 4*(x+y**2)*y
    return np.array([gradx, grady])

def himmelblau_grad2(x, y):
    return 4*(-17 + x + 3*x**2 + y + 3*y**2) + 1e-6

def rosenbrock(x, y):
    a = 1
    b = 100
    return np.square(a-x) + b*np.square(y-np.square(x))

def rosenbrock_grad(x, y):
    a = 1
    b = 100
    gradx = -2*(a-x) -4*b*(y-x**2)*x
    grady = 2*b*(y-x**2)
    return np.array([gradx, grady])

def rosenbrock_grad2(x, y):
    return 202 +  1200*x**2 - 400*y + 1e-6

def GD(grad, x=np.zeros(2, dtype=np.float32), lr=1e-3, max_iter=10000):
    path = []
    x_prev = np.ones(2)*1000

    for i in range(max_iter):
        x = x - lr*grad(x[0], x[1])
        print(f"Iteration {i+1}: x = {x[0]:.4f}, y = {x[1]:.4f}")
        
        if np.linalg.norm(x-x_prev, ord=2) < 1e-4:
            print(f"Converged (took {i+1} iterations)")
            break
        
        x_prev = x.copy()
        path.append(x)
        
    return x, path

def Newton(grad, grad2, x=np.zeros(2, dtype=np.float32), max_iter=100):
    path = []
    x_prev = np.ones(2)*1000

    for i in range(max_iter):
        x = x - (1 / grad2(x[0], x[1])) * grad(x[0], x[1])
        print(f"Iteration {i+1}: x = {x[0]:.4f}, y = {x[1]:.4f}")
        
        if np.linalg.norm(x-x_prev, ord=2) < 1e-4:
            print(f"Converged (took {i+1} iterations)")
            break
        
        x_prev = x.copy()
        path.append(x)
        
    return x, path

def Visualise(func, X, Y, path, algo):
    xg, yg = np.meshgrid(X, Y)
    Z = func(xg, yg)

    path = np.array(path)

    plt.figure(figsize=(8,6))
    cnt = plt.contour(xg, yg, Z, levels=150)
    plt.colorbar(cnt)

    if algo=="gd":
        if len(path) > 0:
            plt.plot(path[:,0], path[:,1], 'r.-', label='GD Path')
            plt.scatter(path[0,0], path[0,1], c='green', label='Start', s=80)
            plt.scatter(path[-1,0], path[-1,1], c='red', label='End', s=80)

        plt.title("Gradient Descent Trajectory")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        if len(path) > 0:
            plt.plot(path[:,0], path[:,1], 'r.-', label='Newton Path')
            plt.scatter(path[0,0], path[0,1], c='green', label='Start', s=80)
            plt.scatter(path[-1,0], path[-1,1], c='red', label='End', s=80)

        plt.title("Newton's Method Trajectory")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()
        

if __name__ == "__main__":
    # Function F
    X = np.linspace(-2, 2, 1000, dtype=np.float32)
    Y = np.linspace(-2, 2, 1000, dtype=np.float32)

    np.random.seed(17)
    x0 = np.random.choice(X)
    y0 = np.random.choice(Y)
    start = np.array([x0, y0])

    x_final, path = GD(grad_f, x=start)
    Visualise(f, X, Y, path, algo="gd")

    x_final, path = Newton(grad_f, grad2_f, x=start)
    Visualise(f, X, Y, path, algo="nt")
    
    # Funciton Himmelblau
    X = np.linspace(-5, 5, 1000, dtype=np.float32)
    Y = np.linspace(-5, 5, 1000, dtype=np.float32)

    np.random.seed(17)
    x0 = np.random.choice(X)
    y0 = np.random.choice(Y)
    start = np.array([x0, y0])

    x_final, path = GD(himmelblau_grad, x=start)
    Visualise(himmelblau, X, Y, path, algo="gd")

    x_final, path = Newton(himmelblau_grad, himmelblau_grad2, x=start)
    Visualise(himmelblau, X, Y, path, algo="nt")

    # Function Rosenbrock
    X = np.linspace(-1, 1, 1000, dtype=np.float32)
    Y = np.linspace(-0.2, 1, 1000, dtype=np.float32)

    np.random.seed(10)
    x0 = np.random.choice(X)
    y0 = np.random.choice(Y)
    start = np.array([x0, y0])

    x_final_r, path_r = GD(rosenbrock_grad, x=start)
    Visualise(rosenbrock, X, Y, path_r, algo="gd")

    x_final_r, path_r = Newton(rosenbrock_grad, rosenbrock_grad2, x=start, max_iter=3000)
    Visualise(rosenbrock, X, Y, path_r, algo="nt")
