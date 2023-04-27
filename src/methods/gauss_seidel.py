# Logan Gregg
# Assignment: MATH 374 Final Project
# Due Date: 5/05/2023
# File Description: File with the serial and parallel 
#                   implementations for the Gauss-Seidel
#                   method for solving linear systems

import numpy as np
from multiprocessing import Pool

def gauss_seidel_serial(A, b, x0, tol, max_iterations):
    n = len(b)
    x = np.copy(x0)

    for iteration in range(max_iterations):
        x_prev = np.copy(x)

        for i in range(n):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x_prev[i+1:])) / A[i, i]

        if np.linalg.norm(x - x_prev) < tol:
            break

    return x

def gauss_seidel_parallel(A, b, x0, tol, max_iterations, num_processes):
    n = len(b)
    x = np.copy(x0)

    def update_equation(i):
        x_new = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        return i, x_new

    for iteration in range(max_iterations):
        x_prev = np.copy(x)

        with Pool(processes=num_processes) as pool:
            results = pool.map(update_equation, range(n))

        for i, x_new in results:
            x[i] = x_new

        if np.linalg.norm(x - x_prev) < tol:
            break

    return x

def main():
    A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
    b = np.array([5, 5, 10])
    
    # Create an initial guess
    x0 = np.zeros_like(b)
    
    # Set the tolerance and max number of iterations
    tolerance = 1e-6
    max_iterations = 1000

    # Use the serial method
    solution = gauss_seidel_serial(A, b, x0, tolerance, max_iterations)
    print(solution)
    
    # Use the parallel method
    num_processes = 4
    solution = gauss_seidel_parallel(A, b, x0, tolerance, max_iterations, num_processes)
    print(solution)
    
if __name__ == "__main__":
    main()
