# HYGO
<img src="logo/HyGO_logo.png" alt="Alt text" title="HyGO - Hybrid Genetic Optimization">

**HYGO** is a versatile, modular toolbox for evolutionary optimization and genetic algorithms. It supports both parametric and control law optimizations and comes equipped with various genetic operations, advanced exploitation methods (such as Downhill Simplex and CMA-ES), and extensive parameter management. The toolbox is designed for researchers and engineers who need to solve complex optimization problems and experiment with evolutionary strategies.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Running a Control Law Optimization](#running-a-control-law-optimization)
  - [Running a Parametric (Rosenbrock) Optimization](#running-a-parametric-rosenbrock-optimization)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Modular Design:** Easy-to-extend components for different optimization tasks.
- **Multiple Optimization Types:** Supports both parametric (e.g., Rosenbrock) and control law optimizations.
- **Genetic Operations:** Implements replication, mutation, crossover, and advanced selection strategies.
- **Exploitation Methods:** Incorporates Downhill Simplex and CMA-ES algorithms for local search and exploitation.
- **Parameter Handling:** Converts between real parameters and binary chromosome representations.
- **Customization:** Provides flexibility with custom rounding functions, parameter scaling, and validity checks.
- **Examples Included:** Ready-to-run example scripts and parameter sets for different problem types.

---

## Project Structure

```plaintext
HYGO/
├── examples/                  # Example scripts and parameter definitions for running optimizations
│   ├── example_ControlLaw.py  # Example for control law optimization
│   ├── example_Parametric.py  # Example for parametric (Rosenbrock) optimization
│   ├── parameters_control_law.py
│   ├── parameters_control_law_Lorentz.py
│   └── parameters_Rosenbrock.py
├── hygo/                      # Core package of the toolbox
│   ├── __init__.py            # Package initializer
│   ├── HYGO.py                # Main high-level class to run the optimization process
│   ├── individual.py          # Defines the individual representation for the GA
│   ├── table.py               # Implements the population table for tracking individuals
│   └── tools/                 # Collection of utility modules and genetic operators
│       ├── choose_operation.py        # Selects genetic operations based on given probabilities
│       ├── CMA_ES.py                  # Implements the CMA-ES exploitation algorithm
│       ├── chromosome_to_params.py    # Converts a binary chromosome to real parameter values
│       ├── DummyParams.py             # Dummy parameters for testing/loading data
│       ├── findKClosest.py            # Finds the K closest individuals in parameter space
│       ├── individual_forced_generator.py  # Generates forced individuals for initialization
│       ├── operations.py              # Contains definitions for genetic operations and their expressions
│       ├── Parameter_help.txt         # Text file documenting parameter options and descriptions
│       ├── parse_parameter_help.py    # Parses the parameter help text into a dictionary
│       ├── params_to_chromosome.py    # Converts parameter lists into a binary chromosome
│       ├── round_params.py            # Rounds parameters using custom functions if provided
│       ├── select_individual.py       # Implements tournament selection for individuals
│       └── simplex.py                 # Implements the Downhill Simplex exploitation algorithm
├── LICENSE                      # License file (e.g., MIT License)
├── Makefile                     # Optional make commands for building/testing
├── pyproject.toml               # Python project configuration
├── setup.py                     # Installation and packaging script
└── README.md                    # Project documentation (this file)
```

## Installation

### Employing PyPi

Install the module into your python installation or virtual environment

```
pip install hygo
``` 

### Cloning the repository

1. **Clone the Repository:**

```
git clone https://github.com/ipatazas/HYGO.git
cd HYGO
``` 

2. **Install Dependencies:**

Ensure you have Python installed. Then run:

```
pip install -r HYGO.egg-info/requires.txt
```

Alternatively, you can install the package in editable mode:

```
pip install -e .
```

## Dependencies

The modules required that will be automatically installed upon employing ```pip install hygo``` are:

```
numpy<2.0,>=1.16
scipy>=1.15,>=1.3
matplotlib>=3.0
pandas>=0.25
dill>=0.3.0
```

The optional packages employed for plotting are:

```
networkx
seaborn
scikit-learn
```

## Usage

### Running a Control Law Optimization
This example uses custom parameters defined for control law optimization.

1. Run the example script:

    ```
    python examples/example_ControlLaw.py
    ```

2. What it does:

    - Initializes the HYGO object with control law parameters from parameters_control_law.py.

    - Executes the genetic algorithm process, including exploitation and parameter tuning.

    - Saves the results and plots the convergence using the custom plotter function.

### Running a Parametric (Rosenbrock) Optimization
This example is set up for a parametric problem using the Rosenbrock function.

1. Run the example script:

    ```
    python examples/example_Parametric.py
    ```

2. What it does:

    - Loads parameters from parameters_Rosenbrock.py.

    - Creates the HYGO object and runs the optimization process.

    - Saves the results and visualizes convergence and cost landscapes using the provided plotter.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
