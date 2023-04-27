import numpy as np
import json
import os

def generate_linear_system(n):
    A = np.random.rand(n, n)  # Coefficient matrix
    b = np.random.rand(n)     # Solution vector
    return A, b

cwd = os.getcwd()

dimensions = [10, 50, 500, 2000, 5000]

for n in dimensions:
    A, b = generate_linear_system(n)

    # Convert arrays to lists for serialization
    A_list = A.tolist()
    b_list = b.tolist()

    # Create a dictionary to hold the data
    data = {"dimension": n, "coefficient_matrix": A_list, "solution_vector": b_list}

    # Serialize data to JSON
    serialized_data = json.dumps(data, indent=4)

    # Write JSON data to a file
    filename = f"linear_system_{n}.json"
    with open("array_data.json", "w") as file:
        file.write(serialized_data)

    print(f"Dimension: {n}")
    print(f"Data written to: {filename}")
    print("-----------------------------------------")