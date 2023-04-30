import numpy as np
import json
from scipy.sparse import random

def generate_linear_system(dimension):
    density = 0.1  # Adjust the density to control sparsity
    min_condition_number = 10  # Minimum condition number for non-singularity
    while True:
        coefficient_matrix = random(dimension, dimension, density=density)
        condition_number = np.linalg.cond(coefficient_matrix.toarray())
        if condition_number >= min_condition_number:
            break
    solution_vector = np.random.rand(dimension)
    linear_system = {
        "dimension": dimension,
        "coefficient_matrix": coefficient_matrix.toarray().tolist(),
        "solution_vector": solution_vector.tolist()
    }
    return linear_system

def main():
    # Dimensions for the systems of linear equations
    dimensions = [25, 100, 250, 500, 750, 1000, 1250, 1400, 1600, 1800, 2000]

    # Generate linear systems and store them in a list
    linear_systems = []
    for dimension in dimensions:
        linear_system = generate_linear_system(dimension)
        linear_systems.append(linear_system)

    # Save the linear systems to a JSON file
    output_file = 'sparse_linear_systems.json'
    with open(output_file, 'w') as file:
        json.dump(linear_systems, file, indent=4)

    print("Linear systems saved to", output_file)

if __name__ == "__main__":
    main()
