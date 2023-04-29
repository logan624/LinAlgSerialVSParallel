# Assignment: MATH 374 Final Project
# Due Date: 5/05/2023
# File Description: File with the serial and parallel implementations for the
#                   LU-Factorization method for solving linear systems

import json
import numpy as np
import multiprocessing
import time

def lu_factorization_solve_serial(A, b):
    n = A.shape[0]
    L = np.eye(n)
    U = np.copy(A)

    # LU-factorization
    for k in range(n-1):
        for i in range(k+1, n):
            if U[k, k] != 0:
                L[i, k] = U[i, k] / U[k, k]
            else:
                L[i, k] = 0  # Set a default value of 0
            for j in range(k, n):
                U[i, j] -= L[i, k] * U[k, j]

    # Solve Ly = b using forward substitution
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Solve Ux = y using backward substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if U[i, i] != 0:
            x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
        else:
            x[i] = 0  # Set a default value of 0

    return x

def update_rows(args):
    k, L, U = args
    n = L.shape[0]
    
    for i in range(k + 1, n):
        if U[k, k] != 0:
            L[i, k] = U[i, k] / U[k, k]
        else:
            L[i, k] = 0  # Set a default value of 0

        for j in range(k, n):
            U[i, j] -= L[i, k] * U[k, j]

def lu_factorization_solve_parallel(A, b, num_processes):
    n = A.shape[0]
    L = np.eye(n)
    U = np.copy(A)

    # LU-factorization
    for k in range(n-1):
        # Create a pool of worker processes
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Prepare arguments for parallel updates
            args = [(k, L, U) for _ in range(n-k-1)]
            # Perform parallel updates for each row
            pool.map(update_rows, args)

    # Solve Ly = b using forward substitution
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Solve Ux = y using backward substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if U[i, i] != 0:
            x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
        else:
            x[i] = 0  # Set a default value of 0

    return x

def deserialize_linear_systems(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        linear_systems = []
        
        for linear_system in data:
            dimension = linear_system['dimension']
            coefficient_matrix = np.array(linear_system['coefficient_matrix'])
            solution_vector = np.array(linear_system['solution_vector'])
            deserialized_system = {
                'dimension': dimension,
                'coefficient_matrix': coefficient_matrix,
                'solution_vector': solution_vector
            }
            linear_systems.append(deserialized_system)
        
        return linear_systems

def main():
    # JSON file path
    file_path = 'linear_systems.json'

    deserialized_systems = deserialize_linear_systems(file_path)
    with open("lu_fact.csv", 'w') as csv_file:
        csv_file.write("dimension, Serial LU Time, Parallel LU Time\n");
        for i in range(2):
            # Accessing the deserialized linear systems
            for system in deserialized_systems:
                dimension = system['dimension']
                coefficient_matrix = system['coefficient_matrix']
                solution_vector = system['solution_vector']
                
                A = coefficient_matrix
                b = solution_vector
                
                print("For n = " + str(dimension) + ":")
                
                # Use the serial method 
                start_time = time.time();
                x = lu_factorization_solve_serial(A, b)
                end_time = time.time();
                serial_time_elapsed = end_time - start_time;
                
                print("\tSerial Time Elapsed: " + str(serial_time_elapsed) + " s")
                
                if dimension <= 250:
                    # Use the parallel method
                    num_processes = multiprocessing.cpu_count()  # Set the number of parallel processes
                    start_time = time.time()
                    x = lu_factorization_solve_parallel(A, b, 2)
                    end_time = time.time();
                    parallel_time_elapsed = end_time - start_time;
                
                    print("\tParallel Time Elapsed: " + str(parallel_time_elapsed) + " s")
                    csv_file.write(str(dimension) + "," + str(serial_time_elapsed) + "," + str(parallel_time_elapsed) + "\n")
                else:
                    print("\tParallel Time Elapsed: N/A")
                    csv_file.write(str(dimension) + "," + str(serial_time_elapsed) + "," + str(0) + "\n")
                                
        csv_file.close()
    
if __name__ == "__main__":
    main()
