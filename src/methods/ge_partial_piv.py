# Logan Gregg
# Assignment: MATH 374 Final Project
# Due Date: 5/05/2023
# File Description: File with the serial and parallel implementations for the
#                   Gaussian Elimination with Partial Pivoting method for solving
#                   linear systems

import numpy as np
from multiprocessing import Pool

def gaussian_elimination_serial(A, b):
    n = len(b)
    Ab = np.concatenate((A, np.expand_dims(b, axis=1)), axis=1)

    # Forward elimination
    for k in range(n-1):
        # Partial pivoting
        max_index = np.argmax(np.abs(Ab[k:, k])) + k
        Ab[[k, max_index]] = Ab[[max_index, k]]

        for i in range(k+1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= factor * Ab[k, k:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:-1], x[i+1:])) / Ab[i, i]

    return x

def gaussian_elimination_parallel(A, b, num_processes):
    n = len(b)
    Ab = np.concatenate((A, np.expand_dims(b, axis=1)), axis=1)

    def forward_elimination(k):
        max_index = np.argmax(np.abs(Ab[k:, k])) + k
        Ab[[k, max_index]] = Ab[[max_index, k]]

        for i in range(k+1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= factor * Ab[k, k:]

    # Forward elimination
    with Pool(processes=num_processes) as pool:
        pool.map(forward_elimination, range(n-1))

    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:-1], x[i+1:])) / Ab[i, i]

    return x

def main():
    A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    b = np.array([1, 0, 1])
    
    # Use the serial method 
    x = gaussian_elimination_serial(A, b)
    print(x)
    
    # Use the parallel method
    num_processes = 4  # Set the number of parallel processes
    x = gaussian_elimination_parallel(A, b, num_processes)
    print(x)
    
if __name__ == "__main__":
    main()
