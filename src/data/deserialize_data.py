import numpy as np
import json

def deserialize_linear_system(filename):
    with open("array_data.json", "r") as file:
        serialized_data = file.read()

    # Deserialize JSON data
    data = json.loads(serialized_data)

    # Retrieve dimension, coefficient matrix, and solution vector
    dimension = data["dimension"]
    A_list = data["coefficient_matrix"]
    b_list = data["solution_vector"]

    # Convert lists back to NumPy arrays
    A = np.array(A_list)
    b = np.array(b_list)

    return dimension, A, b

# Example usage
filename = "linear_system_10.json"
dimension, A, b = deserialize_linear_system(filename)

print(f"Dimension: {dimension}")
print("Coefficient Matrix (A):")
print(A)
print("\nSolution Vector (b):")
print(b)