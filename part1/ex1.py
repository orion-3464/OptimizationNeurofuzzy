import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return np.square(x) + np.square(y-1) + np.power(x-y, 4)

def grad_f(x, y):
    gradx = 2*x + 4*(x-y)**3
    grady = 2*(y-1) - 4*(x-y)**3
    return np.array([gradx, grady])

def f_hessian(x, y):
    term = 12 * (x - y)**2
    fxx = 2 + term
    fyy = 2 + term
    fxy = -term
    return np.array([[fxx, fxy], 
                     [fxy, fyy]])
    
def himmelblau(x, y):
    return np.square(np.square(x)+y-11) + np.square(x+np.square(y))

def himmelblau_grad(x, y):
    gradx = 4*(x**2+y-11)*x + 2*(x+y**2)
    grady = 2*(x**2+y-11) + 4*(x+y**2)*y
    return np.array([gradx, grady])

def himmelblau_hessian(x, y):
    fxx = 12*x**2 + 4*y - 42
    fyy = 12*y**2 + 4*x - 26
    fxy = 4*(x + y)
    return np.array([[fxx, fxy], 
                     [fxy, fyy]])

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

def rosenbrock_hessian(x, y):
    fxx = 2 - 400*(y - x**2) + 800*x**2
    fyy = 200
    fxy = -400*x
    return np.array([[fxx, fxy], 
                     [fxy, fyy]])

def GD(grad, x=np.zeros(2, dtype=np.float32), lr=1e-3, max_iter=10000):
    x = x.copy()
    path = [x]
    x_prev = np.ones(2)*1000

    for i in range(max_iter):
        x = x - lr*grad(x[0], x[1])
        #print(f"Iteration {i+1}: x = {x[0]:.4f}, y = {x[1]:.4f}")
        
        if np.linalg.norm(x-x_prev, ord=2) < 1e-4:
            print(f"Converged (took {i+1} iterations)")
            break
        
        x_prev = x.copy()
        path.append(x)
        
    return x, path

def Newton(grad, hessian, x=np.zeros(2, dtype=np.float32), max_iter=100):
    x = x.copy()
    path = [x]
    x_prev = np.ones(2)*1000

    for i in range(max_iter):
        g = grad(x[0], x[1])
        H = hessian(x[0], x[1])
        
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

    x_final, path = Newton(grad_f, f_hessian, x=start)
    Visualise(f, X, Y, path, algo="nt")
    
    # Funciton Himmelblau
    X = np.linspace(-7, 7, 1000, dtype=np.float32)
    Y = np.linspace(-7, 7, 1000, dtype=np.float32)

    np.random.seed(17)
    x0 = np.random.choice(X)
    y0 = np.random.choice(Y)
    start = np.array([x0, y0])

    x_final, path = GD(himmelblau_grad, x=start)
    Visualise(himmelblau, X, Y, path, algo="gd")

    x_final, path = Newton(himmelblau_grad, himmelblau_hessian, x=start)
    Visualise(himmelblau, X, Y, path, algo="nt")

    # Function Rosenbrock
    X = np.linspace(-3, 3, 1000, dtype=np.float32)
    Y = np.linspace(-3, 3, 1000, dtype=np.float32)

    np.random.seed(0)
    x0 = np.random.choice(X)
    y0 = np.random.choice(Y)
    start = np.array([x0, y0])

    x_final_r, path_r = GD(rosenbrock_grad, x=start)
    Visualise(rosenbrock, X, Y, path_r, algo="gd")

    x_final_r, path_r = Newton(rosenbrock_grad, rosenbrock_hessian, x=start, max_iter=3000)
    Visualise(rosenbrock, X, Y, path_r, algo="nt")
