from geneticUAV import *
from map_plots import *

# Set the parameter values
population_size: int = 100
num_parents: int = 10
offspring_size = population_size - num_parents
max_generations: int = 100


# Main genetic algorithm
def genetic_algorithm(_population_size, _num_parents, _offspring_size, _max_generations):
    # Create the initial population

    population = create_population(_population_size)

    # Evolution loop
    generation_count = 0
    while not termination_criteria(generation_count, _max_generations, population):
        # Evolve the population
        population = evolve_population(population, _num_parents, _offspring_size)
        # Increment generation count
        generation_count += 1
    return population


if __name__ == '__main__':
    # Run the genetic algorithm
    population_obtained = genetic_algorithm(population_size, num_parents, offspring_size, max_generations)
    fitness_np = np.array(calculate_fitness(population_obtained))
    print(np.average(calculate_fitness(population_obtained)))
    print(fitness_np.max())
    maxIndex = fitness_np.argmax()
    print(maxIndex)
    print(population_obtained[maxIndex])
    print(population_obtained)
    print(len(population_obtained))
    print(len(population_obtained[maxIndex]))
    plt.plot(fitness_np)
    plt.show()
    circles = convert_points_to_circles(population_obtained[maxIndex], 10)
    plot_polygon_and_circles(exploration_zone, circles)
