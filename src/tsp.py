# Imports
import time
import matplotlib.pyplot as plt
import random
import numpy as np
import math

# Load TSP Data
def read_tsp_file(filename):
    """
    Reads a TSP file and extracts city coordinates.

    Parameters:
    filename (str): Path to the TSP file.

    Returns:
    dict: A dictionary where keys are city IDs and values are (x, y) coordinates.
    """
    cities = {}
    with open(filename, 'r') as file:
        node_section = False
        for line in file:
            line = line.strip()
            if line.startswith("NODE_COORD_SECTION"):
                node_section = True
                continue
            if line == "EOF":
                break
            if node_section:
                parts = line.split()
                if len(parts) == 3:
                    try:
                        city_id = int(parts[0])
                        x, y = map(float, parts[1:])
                        cities[city_id] = (x, y)
                    except ValueError:
                        print(f"Skipping invalid line: {line}")
    return cities

def calculate_distance_matrix(cities):
    """
    Computes the distance matrix for a given set of cities.

    Parameters:
    cities (dict): A dictionary where keys are city IDs and values are (x, y) coordinates.

    Returns:
    np.ndarray: A 2D array representing distances between each pair of cities.
    """
    num_cities = len(cities)
    distance_matrix = np.zeros((num_cities, num_cities))
    city_ids = list(cities.keys())  # List of city IDs for indexing
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:  # No need to compute distance from a city to itself
                x1, y1 = cities[city_ids[i]]
                x2, y2 = cities[city_ids[j]]
                distance_matrix[i, j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  # Euclidean distance
    return distance_matrix

# Parameters
population_size = 250
mutation_rate = 0.2
crossover_rate = 0.8
generations = 100  # Increased generations for better convergence
tournament_size = 5

# 1. Initialize Population with actual node IDs
def random_genome(length):
    """
    Creates a random genome of city indices from 0 to length-1 (inclusive),
    where each number represents a city index.
    """
    # Create a list of city indices (from 0 to length-1)
    numbers = list(range(length))  # City indices should be from 0 to length-1
    random.shuffle(numbers)  # Shuffle to generate a random route
    return numbers

def init_population(population_size, genome_length):
    """
    Creates a population of random genomes (routes) for the TSP problem.

    Parameters:
    population_size (int): The number of individuals (routes) in the population.
    genome_length (int): The number of cities (length of each route).

    Returns:
    list: A population where each individual is a list of city indices.
    """
    return [random_genome(genome_length) for _ in range(population_size)]

# 2. Fitness Function
def fitness(individual, distance_matrix):
    """
    Calculates the fitness of a given individual (route) based on the total distance traveled.
    The shorter the distance, the higher the fitness.

    Parameters:
    individual (list): A list of city indices representing the route.
    distance_matrix (np.ndarray): A 2D array representing the distances between cities.

    Returns:
    float: The fitness value of the individual (shorter distance = higher fitness).
    """
    total_distance = 0
    # Calculate the total distance for the route
    for i in range(len(individual) - 1):
        city1 = individual[i]
        city2 = individual[i + 1]
        total_distance += distance_matrix[city1, city2]

    # Add distance from the last city back to the first one to make it a closed loop
    total_distance += distance_matrix[individual[-1], individual[0]]

    return total_distance  # Return the total distance itself

def fitness_population(population, distance_matrix):
    """
    Calculates the fitness for the entire population of individuals (routes).

    Parameters:
    population (list): A list of individuals (routes) represented by city indices.
    distance_matrix (np.ndarray): A 2D array representing the distances between cities.

    Returns:
    list: A list of fitness values corresponding to each individual in the population.
    """
    return [fitness(individual, distance_matrix) for individual in population]

# 3. Tournament Selection
def tournament_selection(population, fitness_values):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(enumerate(fitness_values)), tournament_size)
        winner = min(tournament, key=lambda x: x[1])[0]  # Select best
        selected.append(population[winner])
    return selected

# 4. Order Crossover
# OX
def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]
    remaining = [city for city in parent2 if city not in child]

    index = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = remaining[index]
            index += 1
    return child

# PMX
def partially_mapped_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child = [-1] * size
    child[start:end] = parent1[start:end]
    mapping = {parent1[i]: parent2[i] for i in range(start, end)}
    for i in range(size):
        if child[i] == -1:
            city = parent2[i]
            # Resolve conflicts using mapping
            while city in mapping:
                city = mapping[city]

            child[i] = city

    return child

# 5. Mutation Operator
# Swap
def swap_mutation(individual):
    i, j = random.sample(range(len(individual)), 2)
    individual[i], individual[j] = individual[j], individual[i]
    return individual

# Inversion
def inversion_mutation(individual):
    i, j = sorted(random.sample(range(len(individual)), 2))
    individual[i:j+1] = reversed(individual[i:j+1])
    return individual

# Stopping criteria
def stop_function(no_improvement, gens):
    return no_improvement >= gens

def genetic_algorithm(filepath, generations, population_size, mutation_rate, crossover_rate, tournament_size):
    start_time = time.time()
    cities = read_tsp_file(filepath)
    distance_matrix = calculate_distance_matrix(cities)
    population = init_population(population_size, len(cities))

    best_solution = None
    best_fitness = float('inf')
    fitness_history = []
    average_fitness_history = []

    # fitness plateau
    gens = 20
    no_improvement = 0

    for generation in range(generations):
        fitness_values = fitness_population(population, distance_matrix)

        # Update best solution
        min_fitness = min(fitness_values)
        fitness_history.append(min_fitness)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_solution = population[fitness_values.index(min_fitness)]
            no_improvement = 0  # reset counter
        else:
            no_improvement += 1  # increase counter

        # Print the best fitness for the current generation
        print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}")

        if stop_function(no_improvement, gens):
            print("Stopping criteria reached, no improvement for 10 consecutive generations")
            break

        # Selection
        selected_population = tournament_selection(population, fitness_values)

        # Crossover
        offspring = []
        for i in range(0, len(selected_population) - 1, 2):  # Ensure valid pairs
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            if random.random() < crossover_rate:
                child1 = order_crossover(parent1, parent2)
                child2 = order_crossover(parent2, parent1)
            else:
                child1, child2 = parent1[:], parent2[:]  # Copy parents if no crossover
            offspring.extend([child1, child2])

        # Mutation
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] = swap_mutation(offspring[i])

        population = offspring

    # Final results
    elapsed_time = time.time() - start_time
    average_fitness = np.mean(fitness_history)
    average_fitness_history.append(average_fitness)

    print("\nFinal Best Path:", best_solution)
    print(f"Final Best Distance (Fitness): {best_fitness:.2f}")
    print(f"Average Fitness: {average_fitness:.2f}")
    print(f"Computational Time: {elapsed_time:.2f} seconds")

    # Plot Fitness Convergence
    plt.plot(fitness_history, label="Best Fitness Over Generations")
    plt.xlabel("Generations")
    plt.ylabel("Best Fitness")
    plt.title("Genetic Algorithm Convergence")
    plt.legend()
    plt.show()

    return best_solution, best_fitness, fitness_history, average_fitness_history, elapsed_time

def plot_results(parameter_sets, filepath, generations, tournament_size):
    for i, (population_size, mutation_rate, crossover_rate) in enumerate(parameter_sets):
        best_solution, best_fitness, fitness_history, average_fitness_history, elapsed_time = genetic_algorithm(
            filepath, generations, population_size, mutation_rate, crossover_rate, tournament_size)

        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history, label="Best Fitness", color='blue')
        plt.plot(average_fitness_history, label="Average Fitness", color='orange')
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title(f"Genetic Algorithm Convergence\nPopulation: {population_size}, Mutation Rate: {mutation_rate}, Crossover Rate: {crossover_rate}\nComputational Time: {elapsed_time:.2f} seconds")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    filepath = "./TSP_files/berlin52.tsp"
    generations = 100
    tournament_size = 5
    parameter_sets = [
        (100, 0.02, 0.8),  # Baseline
        (200, 0.02, 0.8),  # Higher population
        (100, 0.05, 0.9)   # Higher mutation and crossover
    ]
    plot_results(parameter_sets, filepath, generations, tournament_size)