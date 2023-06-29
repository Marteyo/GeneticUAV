import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from data import exploration_zone
from shapely.ops import unary_union
import multiprocessing as mp
import random

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


# Create initial individual to put in the population putting the points like a grid
def create_individual_grid(individual_size=100, randomization_percentage=0.0, radius_point=253.3):
    poly = Polygon(exploration_zone)
    bounds = poly.bounds

    # Choose a random vertex as the origin
    random_vertex = random.choice(poly.exterior.coords)
    origin_x, origin_y = random_vertex

    # Calculate the grid spacing based on the individual size
    grid_spacing = int(np.ceil(np.sqrt(individual_size)))

    # Calculate the cell size required to cover the polygon
    cell_width = (bounds[2] - bounds[0]) / grid_spacing
    cell_height = (bounds[3] - bounds[1]) / grid_spacing

    # Calculate the number of cells needed to cover the polygon
    x_cells = int(np.ceil((bounds[2] - origin_x) / cell_width))
    y_cells = int(np.ceil((bounds[3] - origin_y) / cell_height))

    # Adjust the grid spacing to ensure complete coverage
    grid_spacing = max(x_cells, y_cells)

    # Generate grid cell coordinates
    x_coords = np.linspace(bounds[0] + cell_width / 2, bounds[2] - cell_width / 2, grid_spacing)
    y_coords = np.linspace(bounds[1] + cell_height / 2, bounds[3] - cell_height / 2, grid_spacing)

    # Create a meshgrid of the coordinates
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Randomly decide the transformation combinations
    if randomization_percentage > 0.0:
        a = random.random()
        _random = (1 - randomization_percentage) / 2
        if a < randomization_percentage:
            # Add random noise and rotate the grid
            noise_scale = 3.0
            xx += np.random.uniform(-noise_scale, noise_scale, size=xx.shape)
            yy += np.random.uniform(-noise_scale, noise_scale, size=yy.shape)
            grid_angle = np.random.uniform(0, 360)  # Random angle in degrees
            theta = np.radians(grid_angle)
            x_center = (bounds[2] + bounds[0]) / 2
            y_center = (bounds[3] + bounds[1]) / 2
            x_rotated = (xx - x_center) * np.cos(theta) - (yy - y_center) * np.sin(theta) + x_center
            y_rotated = (xx - x_center) * np.sin(theta) + (yy - y_center) * np.cos(theta) + y_center
            xx = x_rotated
            yy = y_rotated
        elif a < randomization_percentage + _random:
            # Add random noise to the grid points
            noise_scale = 3.0
            xx += np.random.uniform(-noise_scale, noise_scale, size=xx.shape)
            yy += np.random.uniform(-noise_scale, noise_scale, size=yy.shape)
        else:
            # Randomly rotate the grid
            grid_angle = np.random.uniform(0, 360)
            theta = np.radians(grid_angle)
            x_center = (bounds[2] + bounds[0]) / 2
            y_center = (bounds[3] + bounds[1]) / 2
            x_rotated = (xx - x_center) * np.cos(theta) - (yy - y_center) * np.sin(theta) + x_center
            y_rotated = (xx - x_center) * np.sin(theta) + (yy - y_center) * np.cos(theta) + y_center
            xx = x_rotated
            yy = y_rotated

    # Flatten the grid cell coordinates
    centers = np.column_stack((xx.flatten(), yy.flatten()))

    # Check if each center point intersects with the exploration zone
    circles = [Point(center).buffer(radius_point) for center in centers]
    intersects = np.array([circle.intersects(poly) for circle in circles])

    # Generate chromosomes for activated points
    active_indices = np.where(intersects)[0]
    active_points = np.column_stack((centers[active_indices], np.ones(len(active_indices))))

    # Pad the remaining slots with inactive points
    inactive_points = np.column_stack((np.zeros(individual_size - len(active_points)),
                                       np.zeros(individual_size - len(active_points)),
                                       np.zeros(individual_size - len(active_points))))

    chromosomes = np.vstack((active_points, inactive_points))

    return chromosomes


def create_chromosome(bounds):
    min_x, min_y, max_x, max_y = bounds
    latitude = np.random.uniform(min_x, max_x)
    longitude = np.random.uniform(min_y, max_y)
    binary_value = np.random.randint(0, 2)
    return [latitude, longitude, binary_value]


def create_population(population_size):
    # Generate a population of individuals
    population = np.array([create_individual_grid() for _ in range(population_size)])
    # Add the individual with the grid
    # population = np.append(population, [create_individual_grid(100, 0.2)], axis=0)
    return population


# Implement the genetic operators: selection, crossover, and mutation
def selection(population, fitness_values, num_parents):
    # Select the indices of the top individuals based on fitness values
    top_indices = np.argsort(fitness_values)[-num_parents:]
    # Retrieve the corresponding parents from the population
    parents = population[top_indices]

    return parents


def crossover_operation(parent1, parent2):
    # Generate a mask for selecting genes from parent1 or parent2
    mask = np.random.choice([True, False], size=len(parent1))

    # Perform crossover using the mask
    offspring = np.where(mask[:, None], parent1, parent2)

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


def calculate_fitness_individual_helper(solution):
    activated_points = solution[solution[:, 2] == 1]
    return calculate_fitness_individual(activated_points)


def calculate_fitness(population):
    fitness_values = np.zeros(len(population))
    for i, solution in enumerate(population):
        activated_points = solution[solution[:, 2] == 1]
        fitness_values[i] = calculate_fitness_individual(activated_points)
    return fitness_values

    # with mp.Pool() as pool:
    #     fitness_values = pool.map(calculate_fitness_individual_helper, population)
    # return np.array(fitness_values)


def calculate_fitness_individual(activated_points, radius_point=253.3):
    covered_area = percentage_polygon(activated_points[:, :2], radius_point, exploration_zone)
    overlap_area = circle_overlap_rate(activated_points[:, :2], radius_point)
    # not_in_zone = not_in_exploration_zone(activated_points[:, :2], radius_point, exploration_zone)
    num_activated_points = np.sum(activated_points[:, 2])
    # tradeoff = 0.1
    # if covered_area < 0.90:
    #     fitness_value = (tradeoff * overlap_area + (
    #             1 - tradeoff) * covered_area) * 1000 - num_activated_points - not_in_zone * 10
    # else:
    #     fitness_value = (tradeoff * overlap_area + (
    #             1 - tradeoff) * covered_area) * 1000 - 100 * num_activated_points - not_in_zone * 100
    # # fitness_value = ((overlap_area + covered_area)/2) * 1000 - num_activated_points
    # fitness_value = (tradeoff * overlap_area + (
    #         1 - tradeoff) * covered_area) * 1000 - num_activated_points - not_in_zone * 10

    if covered_area < 0.96:
        return covered_area
    else:
        return (100 - num_activated_points) * 10 + overlap_area * 5
    # return fitness_value


# Perform the generation evolution
def evolve_population(population, num_parents, offspring_size, mutation_rate):
    # Evaluate fitness
    fitness_values = calculate_fitness(population)

    # Perform selection
    parents = selection(population, fitness_values, num_parents)

    # Perform crossover
    offspring_population = crossover(parents, offspring_size)

    # Perform mutation
    mutated_population = mutation(offspring_population, mutation_rate)

    # Combine parents and mutated offspring population
    new_population = np.concatenate((parents, mutated_population), axis=0)

    return new_population


# Define termination criteria
def termination_criteria(generation_count, max_generations, population):
    # Check if the termination criteria (e.g., maximum generations) have been met
    return generation_count >= max_generations or np.average(calculate_fitness(population)) > 900


def percentage_polygon(points, radius_point, polygon):
    """
    Calculate the percentage of a polygon covered by circles with a given radius
    :param points: Center of the circles
    :param radius_point: Radius of the circles
    :param polygon: Polygon to be covered
    :return: Percentage of the polygon covered by circles
    """
    circles = np.array([Point(point[0], point[1]).buffer(radius_point) for point in points])
    poly = Polygon(polygon)

    total_area = poly.area

    union = poly.intersection(unary_union(circles))

    intersection_area = union.area

    if total_area == 0:
        percentage_covered = 0

    else:
        percentage_covered = (intersection_area / total_area)

    return percentage_covered


def circle_overlap_rate(points, radius_point=253.3):
    """
    Calculate the rate of overlap between circles in a given set of points
    :param points: Center of the circles
    :param radius_point: Radius of the circles
    :return: Rate of overlap between circles
    """
    circles = [Point(point[0], point[1]).buffer(radius_point) for point in points]
    union = unary_union(circles)

    total_area = sum(circle.area for circle in circles)
    intersection_area = union.area

    if total_area == 0:
        overlap_rate = 0
    else:
        overlap_rate = intersection_area / total_area

    return overlap_rate


def not_in_exploration_zone(points, radius_point, polygon, center=True):
    """
    Calculate the number of circles with no overlap with the polygon
    :param center: False if take into account the whole circle, True if take into account only the center
    :param points: Center of the circles
    :param radius_point: Radius of the circles
    :param polygon: Polygon to be covered
    :return: Number of circles with no overlap with the polygon
    """
    poly = Polygon(polygon)
    if center:
        intersection_areas = np.array([Point(point[0], point[1]).within(poly) for point in points])
        num_circles_no_overlap = np.count_nonzero(intersection_areas == False)
    else:
        circles = np.array([Point(point[0], point[1]).buffer(radius_point) for point in points])

        intersection_areas = np.array([poly.intersection(circle).area for circle in circles])
        # Check if the circle is completely inside the polygon
        # inside_polygon = np.array([poly.contains(circle) for circle in circles])
        # numb_circles_completely_inside_polygon = np.count_nonzero(inside_polygon == True)
        num_circles_no_overlap = np.count_nonzero(intersection_areas == 0)

    return num_circles_no_overlap
