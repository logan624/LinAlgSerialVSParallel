# Logan Gregg
# Assignment: MATH 374 Final Project
# Due Date: 5/05/2023
# File Description: File with the serial and parallel implementations for the
#                   Jacobi method for solving linear systems

import numpy as np
from multiprocessing import Pool

def jacobi_serial(A, b, x0, max_iterations, tolerance):
    n = len(b)
    x = np.copy(x0)

    for iteration in range(max_iterations):
        x_prev = np.copy(x)

        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x_prev[j]
            x[i] = (b[i] - sigma) / A[i, i]

        if np.linalg.norm(x - x_prev) < tolerance:
            break

    return x

def jacobi_parallel(A, b, x0, max_iterations, tolerance, num_processes):
    n = len(b)
    x = np.copy(x0)

    def update_equation(i):
        sigma = 0
        for j in range(n):
            if j != i:
                sigma += A[i, j] * x0[j]
        return i, (b[i] - sigma) / A[i, i]

    for iteration in range(max_iterations):
        x_prev = np.copy(x)

        with Pool(processes=num_processes) as pool:
            results = pool.map(update_equation, range(n))

        for i, x_new in results:
            x[i] = x_new

        if np.linalg.norm(x - x_prev) < tolerance:
            break

    return x

def main():
    A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
    b = np.array([5, 5, 10])
    
    # Create a guess
    x0 = np.zeros_like(b)
    
    # Set the max iterations and tolerance
    max_iterations = 1000
    tolerance = 1e-6
    
    # Use the serial implementation
    solution = jacobi_serial(A, b, x0, max_iterations, tolerance)
    print(solution)
    
    # Use the parallel implementation
    num_processes = 4
    
    solution = jacobi_parallel(A, b, x0, max_iterations, tolerance, num_processes)
    print(solution)
    
if __name__ == "__main__":
    main()
