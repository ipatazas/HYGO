from dataclasses import dataclass
import numpy as np

def check_validity(GEGA_params,parameters):
    law = parameters[0]
    if law == '0' or law == '0.0':
        return False

    if '(' not in law and np.random.rand(1)[0]<0.8:
        return False
    
    return True

@dataclass
class Parameters:
    
    optimization = 'Control Law' # Type of optimization: Parametric or Control Law

    name = 'Example' #Name of the experiment REQUIRED
    verbose = True #Display information of the process REQUIRED
    MaxTries = 1000 #Maximum number of operations to create an individual REQUIRED
    plotter = lambda GEGA_params,GEGA = None : Plotter(GEGA_params,GEGA)
    batch_evaluation = False
    batch_size = 1
    validity = check_validity

    #Population parameters
    pop_size = 80 #Individuals per generation, can be a int or a list with the
                    # number of inds per generation REQUIRED
    ngen = 10 #Number of gens REQUIRED
    repetitions = 1 #Number of repetitions for each individual REQUIRED
    uncertainty = 0.05 #Uncertainty threshold REQUIRED
    repeat_indivs_outside_uncertainty = True #Repeat those individuals that are outside
                                             # the uncertainty
    badvalue = 1e36 #Value assigned to non-valid individuals
    
    #function utilized to evaluate a population
    cost_function = lambda GEGA_params,parameters,path = None : cost_function(GEGA_params,parameters,path)
    individual_paths = False #Option that if true, each individual will have an assigned
                            # forlder of the form output/geni/repj/individualk where information
                            # is saved. It is useful if the cost function requires file loading. 
                            # If the individual folders do not exist they will be created. If selected,
                            # the cost function must have it as an input
    security_backup = False # A population object will be saved after each individual is evaluated
                           # to make sure that no information is lost

    #Parameters related to the genetic parameters
    Nb_bits = [] #Bits for the parameters. REQUIRED
    N_params = 0    #Number of parameters REQUIRED
    params_range = [] #Limits for parameters, N_paramsx2 mlist REQUIRED
    custom_parameters = [] #Custom parameter ranges
    
    #Convergence parameters
    check_convergence=False #After each generation convergence will be checked
    check_type = 'Generation_change' #Only interval implemented

    #interval convergence: n discretization above or below global minima
    ninterval = 2
    global_minima = []
    
    #Neval convergence: n number of evaluations
    neval = 5000

    #Relative change convergence parameters
    check_n = 100 #Number of individuals to compare
    threshold_convergence=0.1 #Relative change between min and max
                            # to consider convergence

    #Generation change convergence parameters
    generations_stuck = 5 #Number of individuals to compare
    
    # NEWWWWWWWWWW
    limit_evaluations = True # bool to limit number of evaluations
    
    ################################ CONTROL LAW PARAMS ################################
    from ..hygo.tools.operations import operations
    operations = operations
    precission = 5
    
    control_outputs = 1
    control_inputs = 2
    variable_registers = 1
    constant_resisters = 2
    
    number_operations = 9 # Number of operations implemented, for now 9
    
    Ninstructions_initial_min = 5
    Ninstructions_initial_max = 25
    Minimum_instructions = 2
    Max_instructions = 35
    
    Max_eval_time = 20
    
    N_control_points = 100
    
    Sensor_range = [[-1.5,1.5]]*control_inputs
    Control_range = [[-1000,1000]]*control_outputs
    
    SimplexInterpolation = True # Same as Weighted matrix
    
    #-------------Reconstruction Parameters-------------
    
    reconstruction_neval = 1000
    reconstruction_pop_size = 100
    
    ################################################################################################
    
    #Parameters related to genetic operations
    tournament_size = 7 #Tournament size REQUIRED
    p_tour = 1 #Probability of selecting an individual for an operation REQUIRED

    N_elitism = 1 #Number of individuals that will be transfered by elitism REQUIRED

    crossover_points = 1 #Number of crossover points REQUIRED
    crossover_mix = True #True if the selected bits are opposed for bit selection REQUIRED

    mutation_type = 'classic' #Mutation type REQUIRED
    mutation_rate = 0.05 #Bit mutation probability REQUIRED

    p_replication = 0 #Probability of creating an individual by replication REQUIRED
    p_crossover = 0.55 #Probability of creating an individual by crossover REQUIRED
    p_mutation = 0.45 #Probability of creating an individual by mutation REQUIRED

    #Parameters for population

    remove_duplicates = True #Remove individuals that yield equal parameters REQUIRED
    force_individuals = True #If the individuals created will be random or forced REQUIRED
    initialization = 'Random' #Method of initializating the first population.
                              # random and LatinHypercube are implemented REQUIRED

    #Parameters for Latin Hypercube

    LatinN = 10 #Number of individuals to be created with the LatinHypercube sampling method.
                # Required if LatinHypercube selected

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
    
    # Initialization of control law parameters
    def on_init(self):
        self.registers = ['0' for _ in range(self.control_outputs)]
        self.registers += ['s['+str(i)+']' for i in range(self.control_inputs)]
        self.registers += [str(np.random.rand(1)[0]) for i in range(self.variable_registers)]
        self.registers += [str(np.random.rand(1)[0]) for i in range(self.constant_resisters)]
        
        self.register_size = len(self.registers)
        self.variable_registers = len(self.registers)-self.constant_resisters
        
        self.evaluation_time_sample = np.random.rand(self.N_control_points)*self.Max_eval_time
        
        self.ControlPoints = np.random.rand(self.control_inputs,self.N_control_points)
        
        for i in range(self.control_inputs):
            self.ControlPoints[i,:] = self.ControlPoints[i,:]*(self.Sensor_range[i][1]-self.Sensor_range[i][0]) + self.Sensor_range[i][0]

from ..hygo.individual import *
from scipy.integrate import solve_ivp,cumulative_trapezoid
import time

def cost_function(GEGA_params,parameters,path=None):
    '''
    Dummy cost function, it yields the value of cost function given the parameters of an individual,
    it can also serve to measure any experiment since the parameters of each individual are passed
    and such individual has to be measured

    Must have 2 outputs:
        J(float): overall cost function value
        J_terms(list): all cost values from which the J is derived
    '''
    
    R2 = 1 # growth rate
    omega = 1 # freq
    
    IC = [[1,0],[-1,0],[0,1],[0,-1]] # Initial condition
    #IC = [[1,0]] # Initial condition
    
    # time discretization
    start = 0
    stop = 20
    dt = 0.1

    t_eval = np.arange(start=start, stop=stop, step=dt)
    
    gamma = 0.01 # Gamma value

    f = lambda t,s : eval(parameters[0])
    print(parameters[0])
    
    F = lambda t,s : [[(R2 - s[0]*s[0] - s[1]*s[1])*s[0] - omega*s[1]],
                      [(R2 - s[0]*s[0] - s[1]*s[1])*s[1] + omega*s[0] + f(t,s)]]
    
    
    ja = []
    jb = []
    Ja = []
    Jb = []
    J = []
        
    for i,ic in enumerate(IC):
        try:
            ref = time.time()
            def event(t,y):
                return time.time()-(ref+10)
            event.terminal = True
            sol = solve_ivp(F, [start, stop], ic, t_eval=t_eval, method='LSODA', events=[event])

            if sol.status == 0:
                A1 = sol.y[0,:]
                A2 = sol.y[1,:]

                ja.append(np.array(np.power(A1,2)+np.power(A2,2)))
                Ja.append(cumulative_trapezoid(x=t_eval,y=ja[-1]/len(IC)))

                res = []
                for j,t in enumerate(t_eval):
                    res.append(np.power(f(t,[A1[j],A2[j]]),2))

                jb.append(res)
                Jb.append(cumulative_trapezoid(x=t_eval,y=np.array(jb[-1])/len(IC)))
                
                J.append(Ja[-1][-1] + gamma*Jb[-1][-1])
            else:
                Ja.append(GEGA_params.badvalue)
                Jb.append(GEGA_params.badvalue)
                J.append(GEGA_params.badvalue)
        except:
            Ja.append(GEGA_params.badvalue)
            Jb.append(GEGA_params.badvalue)
            J.append(GEGA_params.badvalue)
    print(J)
    return float(np.sum(J)),[J,Ja,Jb]

import matplotlib.pyplot as plt

def Plotter(GEGA_params,GEGA):
    
    best_indiv = GEGA.population[-1].data.loc[0,'Individuals']
    
    params = GEGA.table.individuals[int(best_indiv)].parameters
    
    R2 = 1 # growth rate
    omega = 1 # freq
    
    IC = [[1,0],[-1,0],[0,1],[0,-1]] # Initial condition
    
    # time discretization
    start = 0
    stop = 100
    dt = 0.01

    t_eval = np.arange(start=start, stop=stop, step=dt)
    
    gamma = 0.01 # Gamma value

    f = []

    f = lambda t,s : eval(params[0])
    
    F = lambda t,s : [[(R2 - s[0]*s[0] - s[1]*s[1])*s[0] - omega*s[1]],
                       [(R2 - s[0]*s[0] - s[1]*s[1])*s[1] + omega*s[0] + f(t,s)]]
    
    sols = []
    succ = []
    ics = []
    actuations = []
    
    for i,ic in enumerate(IC):
        sol = solve_ivp(F, [start, stop], ic, t_eval=t_eval, method='LSODA')
        
        if sol.success:
            A1 = sol.y[0,:]
            A2 = sol.y[1,:]
            res = []
            for j,t in enumerate(t_eval):
                res.append(f(t,[A1[j],A2[j]]))
            actuations.append(res)
            succ.append(True)
            sols.append(sol.y)
            ics.append(ic)
        else:
            succ.append(False)
        
    succ = np.array(succ)
    
    fig = plt.figure(constrained_layout=True)
    fig.suptitle('Solutions')
    
    subfigs = fig.subfigures(nrows=np.sum(succ), ncols=1)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f'Initial condition {ics[row]}')
        
        axs = subfig.subplots(nrows=1, ncols=2)
        
        axs[0].plot(sols[row][0,:],sols[row][1,:])
        axs[0].set_title(f'Phase space')
        axs[0].set_xlabel('s[0]')
        axs[0].set_ylabel('s[1]')
        
        axs[1].plot(t_eval,actuations[row])
        axs[1].set_title(f'Actuation')
        axs[1].set_xlabel('t')
        axs[1].set_ylabel('b')
        
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(np.sum(succ)):
        ax.plot(sols[i][0,:],sols[i][1,:])
    ax.set_xlabel('s[0]')
    ax.set_ylabel('s[1]')
    ax.set_title(params)
    
    plt.show()
        
    
            