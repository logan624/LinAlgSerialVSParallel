# Logan Gregg
# Assignment: MATH 374 Final Project
# Due Date: 5/05/2023
# File Description: File with the serial and parallel implementations for the
#                   Gaussian Elimination with Partial Pivoting method for solving
#                   linear systems

import numpy as np
import json
import multiprocessing
import time
import scipy.sparse
import warnings
from scipy.sparse.linalg import spsolve
import multiprocessing

def gaussian_elimination_serial(A, b):
    n = len(b)
    Ab = np.concatenate((A, np.expand_dims(b, axis=1)), axis=1)

    # Forward elimination
    for k in range(n-1):
        # Partial pivoting
        max_index = np.argmax(np.abs(Ab[k:, k])) + k
        Ab[[k, max_index]] = Ab[[max_index, k]]

        for i in range(k+1, n):
            if Ab[k,k] != 0:
                factor = Ab[i, k] / Ab[k, k]
            else:
                factor = 0

    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        if Ab[i,i] != 0:
            x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:-1], x[i+1:])) / Ab[i, i]
        else:
            x[i] = 0

    return x

def gaussian_elimination_sparse(A, b):
    n = len(b)
    Ab = scipy.sparse.hstack((A, scipy.sparse.csc_matrix(np.expand_dims(b, axis=1)))).tocsr()

    # Forward elimination
    for k in range(n-1):
        # Partial pivoting
        max_index = np.argmax(np.abs(Ab[k:, k].toarray())) + k
        Ab[[k, max_index]] = Ab[[max_index, k]]

        for i in range(k+1, n):
            if Ab[k, k] != 0:
                factor = Ab[i, k] / Ab[k, k]
            else:
                factor = 0

    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        if Ab[i, i] != 0:
            x[i] = (Ab[i, -1] - Ab[i, i+1:-1].dot(x[i+1:])) / Ab[i, i]
        else:
            x[i] = 0

    return x

def forward_elimination_sparse(Ab, k):
    n = Ab.shape[0]
    max_index = np.argmax(np.abs(Ab[k:, k])) + k
    Ab[[k, max_index]] = Ab[[max_index, k]]

    for i in range(k + 1, n):
        factor = Ab[i,k]
        if factor != 0:
            factor = Ab[i, k] / Ab[k, k]
        else:
            factor = .000001
        
        Ab[i, k:] -= factor * Ab[k, k:]

def gaussian_elimination_parallel_sparse(A, b, num_processes):
    n = len(b)
    Ab = np.concatenate((A, np.expand_dims(b, axis=1)), axis=1)

    # Forward elimination
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(forward_elimination_sparse, [(Ab, k) for k in range(n - 1)])

    # Solve the system using spsolve
    x = spsolve(Ab[:, :-1], Ab[:, -1])

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
    warnings.filterwarnings('ignore')
    # JSON file path
    file_path = 'sparse_linear_systems.json'

    deserialized_systems = deserialize_linear_systems(file_path)
    with open("partial_pivot.csv", 'w') as csv_file:
        csv_file.write("dimension, Serial GE PP Time, Parallel GE PP Time\n");
        for i in range(3):
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
                x = gaussian_elimination_sparse(A, b)
                end_time = time.time();
                serial_time_elapsed = end_time - start_time;
                
                print("\tSerial Time Elapsed: " + str(serial_time_elapsed) + " s")
                
                # Use the parallel method
                num_processes = multiprocessing.cpu_count()  # Set the number of parallel processes
                start_time = time.time()
                x = gaussian_elimination_parallel_sparse(A, b, num_processes)
                end_time = time.time();
                parallel_time_elapsed = end_time - start_time;
                
                print("\tParallel Time Elapsed: " + str(parallel_time_elapsed) + " s")
                csv_file.write(str(dimension) + "," + str(serial_time_elapsed) + "," + str(parallel_time_elapsed) + "\n")
                        
        csv_file.close()
    
if __name__ == "__main__":
    main()
