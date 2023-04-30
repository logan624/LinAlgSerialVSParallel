# Logan Gregg
# Assignment: MATH 374 Final Project
# Due Date: 5/05/2023
# File Description: File with the serial and parallel 
#                   implementations for the Gauss-Seidel
#                   method for solving linear systems

import numpy as np
import time
import multiprocessing
import json
import warnings

def gauss_seidel_serial(A, b, x0, tol, max_iterations):
    n = len(b)
    x = np.copy(x0)

    for iteration in range(max_iterations):
        x_prev = np.copy(x)

        for i in range(n):
            if np.isclose(A[i, i], 0.0):
                x[i] = 0.0  # Handle division by zero case
            else:
                x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x_prev[i+1:])) / A[i, i]
        if np.linalg.norm(x - x_prev) < tol:
            break

    return x

def gauss_seidel_parallel(A, b, x, tol, num_iterations, num_processes):
    n = len(x)
    pool = multiprocessing.Pool(processes=num_processes)

    for j in range(num_iterations):
        x_new = pool.starmap(update, [(A, b, x, i) for i in range(n)])
        x_old = np.copy(x)
        x = np.array(x_new)
        
        if np.linalg.norm(x - x_old) < tol:
            break
        
    pool.close()
    pool.join()

    return x

# Parallel Gauss-Seidel helper function
def update(A, b, x, i):
    n = len(x)
    sum1 = np.dot(A[i, :i], x[:i])
    sum2 = np.dot(A[i, i + 1:], x[i + 1:])
    x_new = (b[i] - sum1 - sum2) / A[i, i]
    return x_new
    
# To deserialize the JSON file
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
    
    # Deserialize the JSON file
    deserialized_systems = deserialize_linear_systems(file_path)
    
    with open("gauss_seidel.csv", 'w') as csv_file:
        csv_file.write("dimension, Serial GE PP Time, Parallel GE PP Time\n");
        # Collect data for two different trials
        for i in range(2):
            # Accessing the deserialized linear systems
            for system in deserialized_systems:
                dimension = system['dimension']
                coefficient_matrix = system['coefficient_matrix']
                solution_vector = system['solution_vector']
                
                A = coefficient_matrix
                b = solution_vector
                
                # Create an initial guess
                x0 = np.zeros_like(b)
                
                # Set the tolerance and max number of iterations
                tolerance = 1e-6
                max_iterations = 1000
                
                print("For n = " + str(dimension) + ":")
                
                # Use the serial method 
                start_time = time.time();
                x = gauss_seidel_serial(A, b, x0, tolerance, max_iterations)
                
                end_time = time.time();
                serial_time_elapsed = end_time - start_time;
                
                print("\tSerial Time Elapsed: " + str(serial_time_elapsed) + " s")
                
                if dimension < 750:
                    # Use the parallel method
                    num_processes = 2  # Set the number of parallel processes
                    start_time = time.time()
                    x = gauss_seidel_parallel(A, b, x0, tolerance, max_iterations, num_processes)
                    end_time = time.time();
                    parallel_time_elapsed = end_time - start_time;
                
                    print("\tParallel Time Elapsed: " + str(parallel_time_elapsed) + " s")
                    csv_file.write(str(dimension) + "," + str(serial_time_elapsed) + "," + str(parallel_time_elapsed) + "\n")
                else:
                    print("\tParallel Time Elapsed: N/A")
                    
                    # Record the data in a CSV file
                    csv_file.write(str(dimension) + "," + str(serial_time_elapsed) + "," + str(0) + "\n")
                        
        csv_file.close()
    
if __name__ == "__main__":
    main()
