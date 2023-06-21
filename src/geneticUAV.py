import numpy as np
import random
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from data import exploration_zone

exploration_zone = np.array(exploration_zone)


# Define the chromosome representation
def create_individual(individual_size=100):
    poly = Polygon(exploration_zone)
    chromosomes = []
    for i in range(individual_size):
        chromosomes.append(create_chromosome(poly.bounds))

    return chromosomes


def create_chromosome(bounds):
    min_x, min_y, max_x, max_y = bounds
    latitude = random.uniform(min_x, max_x)
    longitude = random.uniform(min_y, max_y)
    return [latitude, longitude, random.randint(0, 1)]


def create_population(population_size):
    # Generate a population of individuals
    return [create_individual() for _ in range(population_size)]


# Implement the genetic operators: selection, crossover, and mutation
def selection(population, fitness_values, num_parents):
    # Select the best individuals as parents for reproduction based on their fitness values
    parents = []
    for _ in range(num_parents):
        max_fitness_index = fitness_values.index(max(fitness_values))
        parents.append(population[max_fitness_index])
        # Set the fitness value of the selected parent to a very low value
        fitness_values[max_fitness_index] = float('-inf')
    return parents


def crossover_operation(parent1, parent2):
    # Perform uniform crossover between parent1 and parent2
    offspring = []
    for gene_index in range(len(parent1)):
        if random.random() < 0.5:
            offspring.append(parent1[gene_index])
        else:
            offspring.append(parent2[gene_index])
    return offspring


def crossover(parents, num_offsprings):
    # Perform crossover between parents to generate offspring
    offsprings = []
    for _ in range(num_offsprings):
        parent1, parent2 = random.sample(parents, 2)
        # Perform crossover operation (e.g., one-point crossover, two-point crossover, etc.)
        # Your implementation here
        offspring = crossover_operation(parent1, parent2)
        offsprings.append(offspring)
    return offsprings


def mutation_operation(individual, mutation_rate):
    # Mutate individual by randomly altering some genes
    mutated_individual = individual.copy()
    poly = Polygon(exploration_zone)

    for gene_index in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            # Generate a random value for the mutated gene within a valid range
            _generate_r = create_chromosome(poly.bounds)
            mutated_individual[gene_index] = _generate_r
    return mutated_individual


def mutation(offsprings, mutation_rate):
    # Perform mutation on the offsprings
    mutated_offsprings = []
    for offspring in offsprings:
        mutated_offspring = mutation_operation(offspring, mutation_rate)
        mutated_offsprings.append(mutated_offspring)
    return mutated_offsprings


def calculate_fitness(population):
    fitness_values = []
    # Cada Individuo de la poblacion es una solución, cada individuo tiene el formato [X,Y,A].
    # X,Y son las coordenadas y A es una flag de activación, por tanto solo se usarán los círculos que esten activos
    for solution in population:
        activated_points = [point for point in solution if point[2] == 1]
        fitness_values.append(
            -(1 - percentage_polygon(activated_points, 253.3, exploration_zone)) * 10 + len(activated_points))
    return fitness_values


# Perform the generation evolution
def evolve_population(population, num_parents, offspring_size):
    # Evaluate fitness
    fitness_values = calculate_fitness(population)

    # Perform selection
    parents = selection(population, fitness_values, num_parents)

    # Perform crossover
    offspring_population = crossover(parents, offspring_size)

    # Perform mutation
    _mutation_rate = 0.04
    mutated_population = mutation(offspring_population, _mutation_rate)

    # Combine parents and mutated offspring population
    new_population = parents + mutated_population

    return new_population


# Define termination criteria
def termination_criteria(generation_count, max_generations, population):
    # Check if the termination criteria (e.g., maximum generations) have been met
    return generation_count >= max_generations or np.average(calculate_fitness(population)) > 1100


def percentage_polygon(points, radius_point, polygon):
    """
    Calculate the percentage of a polygon covered by circles with a given radius
    :param points: Center of the circles
    :param radius_point: Radius of the circles
    :param polygon: Polygon to be covered
    :return: Percentage of the polygon covered by circles
    """
    covered_area = 0.0

    # Create a circle with the given radius for each point
    circles = [Point(point[0], point[1]).buffer(radius_point) for point in points]

    # Create a polygon object from the polygon coordinates
    poly = Polygon(polygon)

    # Iterate over each circle
    total_area = poly.area
    for k, circle1 in enumerate(circles):
        for circle2 in circles[k + 1:]:
            circle1 = circle1.difference(circle2)
        intersection = poly.intersection(circle1)
        intersection_area = intersection.area

        covered_area += intersection_area

    # Calculate the percentage of the polygon covered by circles
    percentage_covered = (covered_area / total_area) * 100

    return percentage_covered
