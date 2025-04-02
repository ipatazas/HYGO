import numpy as np

def select_individual(HYGO_params,pop_size):
    """
    This function implements a tournament selection process. It randomly selects N_tour individuals from the population,
    sorts them, and then chooses the best individual with a probability of p_tour. If the best individual is not chosen,
    the process continues with the second best and so on until an individual is selected.

    Parameters:
        - HYGO_params: A HYGO_params object containing genetic algorithm parameters.
        - pop_size: The size of the population.

    Returns:
        - int: The index of the selected individual.

    """
    
    tournament_size = HYGO_params.tournament_size

    if tournament_size>pop_size:
        tournament_size=pop_size

    selected = []
    
    # Randomly select tournament_size individuals
    for _ in range(tournament_size):
        selected.append(np.ceil(np.random.rand(1).tolist()[0]*pop_size)-1)

    selected.sort()

    r = np.random.rand(1).tolist()[0]

    idx = 0
    while r>HYGO_params.p_tour:
        idx+=1
        if idx==tournament_size:
            idx=0

    return int(selected[idx])