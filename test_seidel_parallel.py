import numpy as np
from multiprocessing import Pool

def gauss_seidel_parallel(A, b, x0, num_iterations, num_processes):
    n = len(b)
    x = x0.copy()
    processes = Pool(processes=num_processes)
    
    for _ in range(num_iterations):
        results = []
        for i in range(n):
            x_temp = x.copy()
            x_temp[i] = 0.0  # Exclude the current variable from the update
            results.append(processes.apply_async(gauss_seidel_update, args=(A, b, x_temp, i)))
        
        x_new = np.zeros(n)
        for i, result in enumerate(results):
            x_new[i] = result.get()
        
        x = x_new.copy()
    
    processes.close()
    processes.join()
    return x

def gauss_seidel_update(A, b, x, i):
    n = len(b)
    sum1 = np.dot(A[i, :i], x[:i])
    sum2 = np.dot(A[i, i + 1:], x[i + 1:])
    x_new = (b[i] - sum1 - sum2) / A[i, i]
    return x_new

# Example usage:
A = np.array([[4.0, -1.0, 0.0, 0.0],
              [-1.0, 4.0, -1.0, 0.0],
              [0.0, -1.0, 4.0, -1.0],
              [0.0, 0.0, -1.0, 3.0]])
b = np.array([5.0, 5.0, 10.0, 15.0])
x0 = np.zeros(len(b))
num_iterations = 100
num_processes = 4

solution = gauss_seidel_parallel(A, b, x0, num_iterations, num_processes)
print(solution)