import json
import numpy as np

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

# Example usage
file_path = 'linear_systems.json'
deserialized_systems = deserialize_linear_systems(file_path)

# Accessing the deserialized linear systems
for system in deserialized_systems:
    dimension = system['dimension']
    coefficient_matrix = system['coefficient_matrix']
    solution_vector = system['solution_vector']
    
    # Do something with the deserialized linear system
    print("Dimension:", dimension)
    print("Coefficient Matrix:", coefficient_matrix)
    print("Solution Vector:", solution_vector)
    print()
