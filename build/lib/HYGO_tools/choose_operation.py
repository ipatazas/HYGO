__author__ = 'Isaac Robledo Martín'
import numpy as np

def choose_operation(HYGO_params):
    """
    Randomly selects a genetic operation based on the provided probabilities.

    Parameters:
        - HYGO_params: An object containing genetic algorithm parameters, including probabilities for replication, mutation, and crossover.

    Returns:
        - str: The selected genetic operation, which can be 'Replication', 'Mutation', or 'Crossover'.

    Raises:
        - Exception: If the sum of replication, mutation, and crossover probabilities is not equal to 1.

    This function generates a random number 'r' between 0 and 1 and compares it with the cumulative probabilities
    of replication (pr), mutation (pm), and crossover (pc). The operation corresponding to the interval in which 'r' falls
    is selected and returned.

    Example:
    ```
    GA_parameters = GeneticAlgorithmParameters(p_replication=0.3, p_mutation=0.2, p_crossover=0.5)
    operation = choose_operation(GA_parameters)
    ```

    The example above would randomly select a genetic operation based on the provided probabilities in GA_parameters.
    """
    pr = HYGO_params.p_replication
    pm = HYGO_params.p_mutation
    pc = HYGO_params.p_crossover

    if round(pr+pm+pc,5) !=1:
        raise Exception('Genetic operations probabilities wrongly selected')
    
    r = np.random.rand()

    if r<pr:
        return 'Replication'
    elif r<(pr+pm):
        return 'Mutation'
    else:
        return 'Crossover'