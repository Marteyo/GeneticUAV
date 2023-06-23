from geneticUAV import *
from map_plots import *

# Set the parameter values
population_size: int = 100  # The population size
num_parents: int = 10  # The number of parents
offspring_size = population_size - num_parents
max_generations: int = 30  # The maximum number of generations
mutation_rate = 0.05  # The probability of mutation


# Main genetic algorithm
def genetic_algorithm(_population_size=100, _num_parents=10, _offspring_size=90, _max_generations=30,
                      _mutation_rate=0.05):
    # Create the initial population

    population = create_population(_population_size)

    # Evolution loop
    generation_count = 0
    while not termination_criteria(generation_count, _max_generations, population):
        # Evolve the population
        population = evolve_population(population, _num_parents, _offspring_size, _mutation_rate)
        # Increment generation count
        generation_count += 1
    print("Termination criteria met after {} generations".format(generation_count))
    return population


if __name__ == '__main__':
    # Run the genetic algorithm
    population_obtained = genetic_algorithm(population_size, num_parents, offspring_size, max_generations)
    fitness_np = np.array(calculate_fitness(population_obtained))
    print(np.average(fitness_np))
    print(fitness_np.max())
    print(fitness_np)
    maxIndex = fitness_np.argmax()
    print(population_obtained[maxIndex])
    plt.plot(fitness_np)
    plt.show()
    circles = convert_points_to_circles(population_obtained[maxIndex], 253.3)
    plot_polygon_and_circles(exploration_zone, circles)
