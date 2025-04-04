from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def check_validity(GEGA_params,parameters):

    for param in parameters:
        if abs(param)>1:
            return False
    return True

def Rastriguin_cost(GEGA_params,parameters,path=None):
    '''
    Doomy cost function, it yields the value of cost function given the parameters of an individual,
    it can also serve to measure any experiment since the parameters of each individual are passed
    and such individual has to be measured

    Must have 2 outputs:
        J(float): overall cost function value
        J_terms(list): all cost values from which the J is derived
    '''
    values = np.array(parameters)

    if len(values.shape)>1:
        J = 10*values.shape[1]
        for i in range(values.shape[1]):
            J += values[:,i]**2 - 10*np.cos(2*np.pi*values[:,i])
        J = np.sum(np.power(parameters,2),axis=1)
        return J.tolist(), [0]*GEGA_params.batch_size
    else:
        J = 10*values.shape[0]
        for i in range(values.shape[0]):
            J += values[i]**2 - 10*np.cos(2*np.pi*values[i])
        return float(J), 0

def Rastriguin_plotter(GA_params,GA):

    x = np.linspace(GA.parameters.params_range[0][0],GA.parameters.params_range[0][1],1000)
    y = np.linspace(GA.parameters.params_range[1][0],GA.parameters.params_range[1][1],1000)

    xv, yv = np.meshgrid(x, y, indexing='ij')

    zz = np.zeros((1000,1000))

    for i in range(1000):
        for j in range(1000):
            val = Rastriguin_cost(GA_params,[x[i],y[j]],None)
            zz[i,j] = val[0]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    best_idx = []
    best_cost = []
    best_x = []
    best_y = []
    for pop in GA.population:
        best_idx.append(int(pop.data.loc[0,'Individuals']))
        best_cost.append(GA.table.individuals[int(best_idx[-1])].cost)
        best_x.append(GA.table.individuals[int(best_idx[-1])].parameters[0])
        best_y.append(GA.table.individuals[int(best_idx[-1])].parameters[1])
    
    colors=cm.tab10(len(best_cost))

    ax.plot3D(best_x,best_y,best_cost,'r-',linewidth=1.5)
    ax.scatter3D(best_x,best_y,best_cost,s=10,color=colors)

    surf = ax.plot_surface(xv, yv, zz, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False,alpha=0.3)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('J')

@dataclass
class Parameters:
    optimization = 'Parametric' # Type of optimization: Parametric or Control Law

    name = 'Rastriguin' #Name of the experiment REQUIRED
    verbose = True #Display information of the process REQUIRED
    MaxTries = 2 #Maximum number of operations to create an individual REQUIRED
    plotter = Rastriguin_plotter
    batch_evaluation = True
    batch_size = 100 

    #Population parameters
    pop_size = 30 #Individuals per generation, can be a int or a list with the
                    # number of inds per generation REQUIRED
    ngen = 8 #Number of gens REQUIRED
    repetitions = 1 #Number of repetitions for each individual REQUIRED
    badvalue = 1e36 #Value assigned to non-valid individuals

    remove_duplicates = True #Remove individuals that yield equal parameters REQUIRED
    force_individuals = False #If the individuals created will be random or forced REQUIRED
    # validity = check_validity

    #function utilized to evaluate a population
    cost_function = Rastriguin_cost
    individual_paths = False #Option that if true, each individual will have an assigned
                            # folder of the form output/geni/repj/individualk where information
                            # is saved. It is useful if the cost function requires file loading. 
                            # If the individual folders do not exist they will be created.
    security_backup = True # A population object will be saved after each individual is evaluated
                           # to make sure that no information is lost

    #2 DIMENSIONAL PROBLEMS
    #Parameters related to the genetic parameters
    Nb_bits = 12 #Bits for the parameters. REQUIRED
    
    custom_parameters = [] #Custom parameter ranges

    params_range = [[-5,5]]*7
    N_params = 7

    #Convergence parameters
    check_convergence = True #After each generation convergence will be checked
    check_type = 'Generation_change' #interval, Neval, Generation_change and Relative_change implemented

    #Generation change convergence parameters
    generations_stuck = 5 #Number of individuals to compare 

    limit_evaluations = False

    #Parameters related to genetic operations
    tournament_size = 7 #Tournament size REQUIRED
    p_tour = 1 #Probability of selecting an individual for an operation REQUIRED

    N_elitism = 1 #Number of individuals that will be transfered by elitism REQUIRED

    crossover_points = 1 #Number of crossover points REQUIRED #<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    crossover_mix = True #True if the selected bits are opposed for bit selection REQUIRED

    mutation_type = 'classic' #Mutation type classic or at_least_one REQUIRED
    mutation_rate = 0.05 #Bit mutation probability REQUIRED 

    p_replication = 0 #Probability of creating an individual by replication REQUIRED
    p_crossover = 0.55 #Probability of creating an individual by crossover REQUIRED
    p_mutation = 0.45 #Probability of creating an individual by mutation REQUIRED

    
    initialization = 'random' #Method of initializating the first population.
                              # random and LatinHypercube are implemented REQUIRED

    #Exploitation parameters

    exploitation = True #It can be a bool or a list of bools, if a list of bools it must have the
                        # the same length as the number of populations
    MaxSimplexCycles = 100 # Maximum number of cycles REQUIRED
    SimplexSize = 10 #It can be an int or a list of ints containing the number of individuals being
                     # considered for the exploitation REQUIRED
    ExploitationType = 'Downhill Simplex' #Explotation type, only Downhill Simplex available for now
    SimplexPool = 'Population' #Select Population or All if the individuals considered for an explotation
                               # step in a given population take into account the whole individual pool REQUIRED
                               # or just the population pool REQUIRED
    SimplexOffspring = 20 # Can be an integer or a list of integers including the number of individuals
                         # that will be generated through the explotation
                             
    SimplexInitialization = 'BestN' #How the simplex is initialized REQUIRED
                                    #   BestN: The best individuals in the population/table
                                    #   ClosestN: Closest individuals in the population/table
    
    SimplexCycleChecker = 5 #Number of cycles to check if the simplex has been looking in a hyperplane
                            #   REQUIRED
    Simplex_R2 = 0.999 #Threshold to consider that the points are in a hyperplane
                      #     REQUIRED
    Simplex_intervals_movement = 300 #Number of intervals that a new point is introduced
                      # in the simplex is moved from the last centroid. REQUIRED
    Simplex_intervals_random_movement = 100 #Number of intervals that a new point is introduced
                      # in the simplex is moved from the best individual when the simplex is in a cycle. REQUIRED

    reflection_alpha = 1 #Reflection hyperparameter
    expansion_gamma  = 2 #expansion hyperparameter
    contraction_rho  = 0.5 #contraction hyperparameter
    shrinkage_sigma  = 0.5 #shrinkage hyperparameter