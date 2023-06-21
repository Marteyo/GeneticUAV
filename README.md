# UAV Point Exploration with Genetic Algorithm

This project aims to utilize a Genetic Algorithm (GA) to determine the optimal set of points for exploration within a given area using Unmanned Aerial Vehicles (UAVs). The algorithm leverages the principles of natural selection and genetic variation to efficiently navigate and explore the target region.

## Table of Contents
- [Introduction](#introduction)
- [Genetic Algorithm Overview](#genetic-algorithm-overview)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

UAVs have gained significant popularity in various fields, including surveillance, search and rescue operations, environmental monitoring, and more. Efficiently exploring a specific area with multiple UAVs can be a challenging task. This project presents a solution by implementing a Genetic Algorithm to optimize the selection of points for exploration by UAVs.

The Genetic Algorithm considers a set of possible exploration points within the target area and evolves a population of solutions iteratively. By simulating natural selection, the algorithm progressively improves the quality of solutions over generations, ultimately converging towards an optimal set of points for exploration.

## Genetic Algorithm Overview

The Genetic Algorithm implemented in this project follows a standard flow:

1. **Initialization**: A population of candidate solutions is created, each representing a possible set of exploration points for UAVs.
2. **Fitness Evaluation**: Each candidate solution is evaluated based on a fitness function, which quantifies the quality of the exploration points selected.
3. **Selection**: Based on their fitness scores, candidate solutions are selected to become parents for the next generation.
4. **Crossover**: Parents are combined through crossover operations to create offspring with a mix of exploration points from both parents.
5. **Mutation**: Some exploration points within the offspring are subject to random mutations to introduce genetic diversity.
6. **Replacement**: The new generation, including both parents and offspring, replaces the previous population.
7. **Termination**: The algorithm iterates through steps 2 to 6 until a termination criterion is met, such as reaching a maximum number of generations or achieving a satisfactory fitness level.

The iterative process of selection, crossover, mutation, and replacement promotes the exploration of the search space and gradually improves the solutions.

## Project Structure

The project has the following structure:

```
project/
├── src/
│   ├── main.py
│   ├── geneticUAV.py
│   ├── map_plots.py
│   └── data.py
├── README.md
└── requirements.txt
```

- `src/`: This directory contains the source code of the project.
  - `main.py`: Entry point of the project. It initializes the exploration environment and runs the genetic algorithm.
  - `geneticUAV.py`: Implementation of the Genetic Algorithm, including selection, crossover, mutation, and replacement operations.
  - `map_plots.py`: Contains functions for visualizing the exploration environment and the results.
  - `data.py`: Includes classes and functions for handling the data and defining the exploration area.
- `README.md`: The document you are currently reading, providing an overview of the project and usage instructions.
- `requirements.txt`: A list of required Python packages for running the project.

## Usage

To run the project, follow these steps:

1. Ensure you have Python 3.10 installed on your system.
2. Clone the project repository:

```bash
git clone https://github.com/Marteyo/GeneticUAV.git
cd your-project
```

3. Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

4. Modify the necessary parameters and functions in the source code to match your specific

 problem and requirements.
5. Run the main script:

```bash
python src/main.py
```

6. Observe the output, including the progress of the Genetic Algorithm, fitness values, and exploration point sets.
7. Utilize the provided functions in `map_plots.py` to visualize the exploration environment and the results.
8. Analyze the final results and adapt the code as needed for your application.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the project's GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to modify and distribute it according to your needs.
