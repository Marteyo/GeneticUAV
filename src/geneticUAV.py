import numpy as np
import random
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from data import exploration_zone

exploration_zone = np.array(exploration_zone)


# Define the chromosome representation
def create_individual(individual_size=100):
    poly = Polygon(exploration_zone)
    bounds = poly.bounds

    latitudes = np.random.uniform(bounds[0], bounds[2], size=individual_size)
    longitudes = np.random.uniform(bounds[1], bounds[3], size=individual_size)
    binary_values = np.random.randint(0, 2, size=individual_size)

    chromosomes = np.column_stack((latitudes, longitudes, binary_values))
    return chromosomes


def create_chromosome(bounds):
    min_x, min_y, max_x, max_y = bounds
    latitude = np.random.uniform(min_x, max_x)
    longitude = np.random.uniform(min_y, max_y)
    binary_value = np.random.randint(0, 2)
    return [latitude, longitude, binary_value]


def create_population(population_size):
    # Generate a population of individuals
    population = np.repeat([create_individual()], population_size, axis=0)
    return population


# Implement the genetic operators: selection, crossover, and mutation
def selection(population, fitness_values, num_parents):
    # Convert fitness_values to a NumPy array
    # itness_values = np.array(fitness_values)

    # Select the indices of the top individuals based on fitness values
    # top_indices = np.argpartition(-fitness_values, num_parents)[:num_parents]
    top_indices = np.argsort(fitness_values)[-num_parents:]
    # Retrieve the corresponding parents from the population
    parents = population[top_indices]

    return parents


def crossover_operation(parent1, parent2):
    # Convert parent1 and parent2 to NumPy arrays
    # parent1 = np.array(parent1)
    # parent2 = np.array(parent2)

    # Generate a mask for selecting genes from parent1 or parent2
    mask = np.random.choice([True, False], size=len(parent1))

    # Perform crossover using the mask
    offspring = np.where(mask[:, None], parent1, parent2)

    # Convert offspring back to a list
    # offspring = offspring.tolist()

    return offspring


def crossover(parents, num_offsprings):
    # Perform crossover between parents to generate offspring
    offsprings = []
    num_parents = len(parents)
    for _ in range(num_offsprings):
        # Randomly select two distinct parents
        parent_indices = np.random.choice(num_parents, size=2, replace=False)
        parent1, parent2 = parents[parent_indices[0]], parents[parent_indices[1]]

        # Perform crossover operation using crossover_operation function
        offspring = crossover_operation(parent1, parent2)
        offsprings.append(offspring)
    return offsprings


def mutation_operation(individual, mutation_rate):
    # Mutate individual by randomly altering some genes
    mutated_individual = individual.copy()

    # Generate a random mask for gene mutation
    mask = np.random.random(size=len(mutated_individual)) < mutation_rate

    # Apply mutation to the genes selected by the mask
    poly = Polygon(exploration_zone)
    mutated_genes = np.array([create_chromosome(poly.bounds) for _ in range(len(mutated_individual))])
    mutated_individual[mask] = mutated_genes[mask]

    return mutated_individual


def mutation(offsprings, mutation_rate):
    # Perform mutation on the offsprings
    mutated_offsprings = np.stack([mutation_operation(offspring, mutation_rate) for offspring in offsprings])
    return mutated_offsprings


def calculate_fitness(population):
    fitness_values = np.zeros(len(population))
    for i, solution in enumerate(population):
        activated_points = solution[solution[:, 2] == 1]
        fitness_values[i] = calculate_fitness_individual(activated_points)
    return fitness_values


def calculate_fitness_individual(activated_points):
    covered_area = percentage_polygon(activated_points[:, :2], 253.3, exploration_zone)
    num_activated_points = np.sum(activated_points[:, 2])
    fitness_value = -(1 - covered_area) * 10 + num_activated_points
    return fitness_value


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
    new_population = np.concatenate((parents, mutated_population), axis=0)

    return new_population


# Define termination criteria
def termination_criteria(generation_count, max_generations, population):
    # Check if the termination criteria (e.g., maximum generations) have been met
    return generation_count >= max_generations or np.average(calculate_fitness(population)) > 1000


def percentage_polygon(points, radius_point, polygon):
    """
    Calculate the percentage of a polygon covered by circles with a given radius
    :param points: Center of the circles
    :param radius_point: Radius of the circles
    :param polygon: Polygon to be covered
    :return: Percentage of the polygon covered by circles
    """
    covered_area = 0.0
    circles = np.array([Point(point[0], point[1]).buffer(radius_point) for point in points])
    poly = Polygon(polygon)

    total_area = poly.area

    for k in range(len(circles)):
        circle1 = circles[k]
        for circle2 in circles[k + 1:]:
            circle1 = circle1.difference(circle2)
        intersection = poly.intersection(circle1)
        intersection_area = intersection.area
        covered_area += intersection_area

    percentage_covered = (covered_area / total_area) * 100

    return percentage_covered
