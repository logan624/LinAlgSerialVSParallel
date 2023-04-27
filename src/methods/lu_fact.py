# Assignment: MATH 374 Final Project
# Due Date: 5/05/2023
# File Description: File with the serial and parallel implementations for the
#                   LU-Factorization method for solving linear systems

import numpy as np
from multiprocessing import Pool

def lu_factorization_solve_serial(A, b):
    n = A.shape[0]
    L = np.eye(n)
    U = np.copy(A)

    # LU-factorization
    for k in range(n-1):
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            for j in range(k, n):
                U[i, j] -= L[i, k] * U[k, j]

    # Solve Ly = b using forward substitution
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Solve Ux = y using backward substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x

def lu_factorization_solve_serial(A, b):
    n = A.shape[0]
    L = np.eye(n)
    U = np.copy(A)

    # LU-factorization
    for k in range(n-1):
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            for j in range(k, n):
                U[i, j] -= L[i, k] * U[k, j]

    # Solve Ly = b using forward substitution
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Solve Ux = y using backward substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x

def main():
    A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    b = np.array([1, 0, 1])

    # Use the serial method
    x = lu_factorization_solve_serial(A, b)
    print(x)
    
    # Use the parallel method
    x = lu_factorization_solve_serial(A, b)
    print(x)
    
if __name__ == "__main__":
    main()
