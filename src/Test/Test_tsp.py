import unittest
import numpy as np
import random

from src.tsp import random_genome, init_population, fitness, fitness_population, tournament_selection, order_crossover, \
    partially_mapped_crossover, swap_mutation, inversion_mutation, genetic_algorithm


class TestGeneticAlgorithmFunctions(unittest.TestCase):
    def test_random_genome(self):
        length = 10
        genome = random_genome(length)
        self.assertEqual(len(genome), length)
        self.assertEqual(sorted(genome), list(range(length)))  # Ensure all cities are present
        print(f"Test random genome: {genome}")

    def test_init_population(self):
        population_size = 5
        genome_length = 10
        population = init_population(population_size, genome_length)
        self.assertEqual(len(population), population_size)
        print("Test init population ")
        for individual in population:
            self.assertEqual(len(individual), genome_length)
            self.assertEqual(sorted(individual), list(range(genome_length)))  # Ensure valid genomes
            print(individual)

    def test_fitness(self):
        distance_matrix = np.array([
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0]
        ])
        individual = [0, 1, 3, 2]
        actual_distance = fitness(individual, distance_matrix)
        expected_distance = 2 + 4 + 12 + 15  # Corrected path distances
        self.assertEqual(fitness(individual, distance_matrix), expected_distance)
        print(f"Test Fitness: Expected: {expected_distance}, Actual: {actual_distance}\n")

    def test_fitness_population(self):
        distance_matrix = np.array([
            [0, 2, 9, 10],
            [1, 0, 6, 4],
            [15, 7, 0, 8],
            [6, 3, 12, 0]
        ])
        population = [[0, 1, 3, 2], [1, 3, 2, 0]]
        fitness_values = fitness_population(population, distance_matrix)
        self.assertEqual(len(fitness_values), len(population))
        print(f"Test Fitness Population: Expected: {len(population)}, Actual: {len(fitness_values)}\n")

    def test_order_crossover(self):
        parent1 = [0, 1, 2, 3, 4, 5]
        parent2 = [3, 4, 5, 0, 1, 2]
        child = order_crossover(parent1, parent2)
        self.assertEqual(len(child), len(parent1))
        self.assertEqual(sorted(child), sorted(parent1))  # Ensure valid genome
        print(f"Test order crossover: Child genome: {child}\n")

    def test_partially_mapped_crossover(self):
        parent1 = [0, 1, 2, 3, 4, 5]
        parent2 = [3, 4, 5, 0, 1, 2]
        child = partially_mapped_crossover(parent1, parent2)
        self.assertEqual(len(child), len(parent1))
        self.assertEqual(sorted(child), sorted(parent1))
        print(f"Test partially mapped crossover: Child genome: {child}\n")

    def test_swap_mutation(self):
        individual = [0, 1, 2, 3, 4]
        mutated = swap_mutation(individual[:])
        self.assertEqual(sorted(mutated), sorted(individual))
        self.assertNotEqual(mutated, individual)  # Ensure mutation occurred
        print(f"Test swap mutation: Individual: {individual}\n")

    def test_inversion_mutation(self):
        individual = [0, 1, 2, 3, 4]
        mutated = inversion_mutation(individual[:])
        self.assertEqual(sorted(mutated), sorted(individual))
        self.assertNotEqual(mutated, individual)  # Ensure mutation occurred
        print(f"Test inversion mutation: Individual: {individual}\n")

if __name__ == "__main__":
    unittest.main()
