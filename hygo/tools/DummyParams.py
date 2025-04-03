__author__ = 'Isaac Robledo Martín'
from dataclasses import dataclass
@dataclass
class DummyParameters:
    '''
    Dummy class used to load data
    '''
    optimization = None
    name = None
    verbose = None
    MaxTries = None
    plotter = None
    pop_size = None
    ngen = None
    repetitions = None
    uncertainty = None
    repeat_indivs_outside_uncertainty = None
    badvalue = None
    cost_function = None
    individual_paths = None
    security_backup = None
    Nb_bits = None
    N_params = None
    params_range = None
    custom_parameters = None
    check_convergence= None
    check_type = None
    ninterval = None
    neval = None
    global_minima = None
    tournament_size = None
    p_tour = None
    N_elitism = None
    crossover_points = None
    crossover_mix = None
    mutation_type = None
    mutation_rate = None
    p_replication = None
    p_crossover = None
    p_mutation = None
    remove_duplicates = None
    force_individuals = None
    initialization = None
    LatinN = None
    exploitation = None
    SimplexSize = None
    ExploitationType = None
    SimplexPool = None
    SimplexOffspring = None
    SimplexInitialization = None
    reflection_alpha = None
    expansion_gamma  = None
    contraction_rho  = None
    shrinkage_sigma  = None