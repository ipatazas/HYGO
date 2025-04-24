__author__ = 'Isaac Robledo Martín'
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import importlib
import ast

from .table import Table
from .population import Population
from .individual import Individual

from .tools.parse_parameter_help import parse_parameter_file

warnings.filterwarnings("ignore")

def check_parameter_type(parameter,value,valid_types):
    if type(valid_types)==list:
        for valid in valid_types:
            if type(value)==valid:
                return
            if valid=='function' and callable(value):
                return
            if valid=='NoneType' and (value is None):
                return
    else:
        if type(value)==valid_types:
            return
        if valid_types=='function' and callable(value):
                return
        if valid_types=='function' and (value is None):
                return
        
    raise TypeError('The Parameter ' + parameter + f' does not have a valid type. It must be {valid_types}')

def check_parameter_value(parameter,value, valid_options):
    if type(valid_options)==list:
        if value in valid_options:
            return
    else:
        if value == valid_options:
            return
        
    raise TypeError('The Parameter ' + parameter + f' is not valid. It must be one  {valid_options}')

def check_parameter(HYGO_parameters, parameter_names, parameter, valid_types, valid_options=None, default=None, specific_condition=None):
    
    if parameter not in parameter_names:
        if default is None:
            raise NameError('The Parameter ' + parameter + f' must be included')
        else:
            if default=='Empty':
                setattr(HYGO_parameters, parameter, None)
            else:
                setattr(HYGO_parameters, parameter, default)
            print(f'Warning: Parameter {parameter} was not introduced and was set to the default value {default}')
    

    value = getattr(HYGO_parameters, parameter)
    
    check_parameter_type(parameter,value,valid_types)
    if valid_options is not None:
        check_parameter_value(parameter,value, valid_options)
    if specific_condition is not None:
        if type(specific_condition)==list:
            for condition in specific_condition:
                check = eval(condition)
                if not check:
                    raise ValueError(f'Parameter {parameter} violates condition '+condition)
        else:
            check = eval(specific_condition)
            if not check:
                raise ValueError(f'Parameter {parameter} violates condition '+specific_condition)
    
    return HYGO_parameters

def check_conditional_parameter(HYGO_parameters, parameter_names, parent, parent_condition, parameter, valid_types, valid_options=None, default=None, specific_condition=None):
    
    if parent in parameter_names:
        value = getattr(HYGO_parameters,parent)
        if eval(parent_condition):
            HYGO_parameters = check_parameter(HYGO_parameters, parameter_names, parameter, valid_types, valid_options, default, specific_condition)
        
    return HYGO_parameters

class HYGO():
    """
    Genetic Evolutionary Algorithm (HYGO) Class

    This class represents a genetic algorithm for optimization problems.
    It is designed to evolve a population of individuals through generations,
    applying genetic operators such as selection, crossover, and mutation. It also features 
    an enrichment process through an exploitation.

    Attributes:
        - generation (int): Current generation number.
        - table (Table): Table object to store and manage individuals' information.
        - population (list): List of Population objects representing each generation.
        - parameters (object): Plant parameters specifying the genetic algorithm settings.
        - output_path (str): Path to the directory for saving outputs.
        - reached_convergence (bool): Flag indicating if the algorithm reached convergence.

    Methods:
        - generate_population(): Generates the initial population.
        - evaluate_population(ngen=None): Evaluates the fitness of individuals in the current or specified generation.
        - evolve_population(): Evolves the population to the next generation using genetic operators.
        - go(ngen=None): Runs the genetic algorithm for the specified number of generations.
        - exploitation(gen,checker,convergence): Method used to perform the exploitation.
        - display_convergence(): Displays information about the best individual and its parameters.
        - check_convergence(): Checks for convergence based on specified criteria.
        - save(path=None, specific_save=False): Saves the current state of the genetic algorithm.
        - load_security(path): Loads the genetic algorithm state after an interruption.
        - load(path, specific_save=False): Loads a previously saved state of the genetic algorithm.
        - convergence(fitness=False, save=None): Visualizes the convergence of the genetic algorithm.
        - plot_gens(gens=0, save=None): Visualizes the individuals' distribution in parameter space for specified generations.

    Note: Ensure the Population, Table, and Individual classes are available for the HYGO class to utilize.
    """

    REQUIRED_PARAMETERS = [
            {'parameter':'optimization','type':str,'valid_options':['Parametric','Control Law']},
            {'parameter':'name','type':str,'default':'HyGO Optimization'},
            {'parameter':'verbose','type':bool,'default':True},
            {'parameter':'MaxTries','type':int,'default':100,'specific':'value>0'},
            {'parameter':'batch_evaluation','type':bool,'default':False},
            {'parameter':'pop_size','type':[int,list,np.ndarray]},
            {'parameter':'ngen','type':int},
            {'parameter':'repetitions','type':int,'default':1,'specific':'value>0'},
            {'parameter':'badvalue','type':[float,int],'default':1e36,'specific':'value>0'},
            {'parameter':'remove_duplicates','type':bool,'default':True},
            {'parameter':'force_individuals','type':bool,'default':False},
            {'parameter':'cost_function','type':'function'},
            {'parameter':'plotter','type':['function','NoneType'], 'default':'Empty'},
            {'parameter':'individual_paths','type':bool,'default':False},
            {'parameter':'security_backup','type':bool,'default':True},
            {'parameter':'Nb_bits','type':[int,list]},
            {'parameter':'custom_parameters','type':list,'default':[]},
            {'parameter':'params_range','type':list},
            {'parameter':'N_params','type':int},
            {'parameter':'check_convergence','type':bool,'default':True},
            {'parameter':'check_type','type':str,'default':'Generation_change','valid_options':['interval', 'Neval', 'Generation_change', 'Relative_change']},
            {'parameter':'limit_evaluations','type':bool,'default':False},
            {'parameter':'tournament_size','type':int},
            {'parameter':'p_tour','type':[int,float],'default':1,'specific':['value<=1','value>0']},
            {'parameter':'N_elitism','type':int,'default':1},
            {'parameter':'crossover_points','type':int,'default':1},
            {'parameter':'crossover_mix','type':bool,'default':True},
            {'parameter':'mutation_type','type':str,'default':'at_least_one','valid_options':['classic','at_least_one']},
            {'parameter':'p_replication','type':[int,float],'default':0.0,'specific':['value<=1','value>=0']},
            {'parameter':'p_crossover','type':[int,float],'default':0.55,'specific':['value<=1','value>=0']},
            {'parameter':'p_mutation','type':[int,float],'default':0.45,'specific':['value<=1','value>=0']},
            {'parameter':'initialization','type':str,'default':'random','valid_options':['random','LatinHypercube']},
            {'parameter':'exploitation','type':bool,'default':False},
        ]
    
    CONDITIONAL_PARAMETERS = [
            {'parameter':'batch_size','type':int,'parent':'batch_evaluation','parent_condition':'value'},
            {'parameter':'repeat_indivs_outside_uncertainty','type':bool,'parent':'repetitions','parent_condition':'value>1'},
            {'parameter':'uncertainty','type':float,'parent':'repetitions','parent_condition':'value>1','specific':'value>0'},
            {'parameter':'N_params','type':int,'parent':'params_range','parent_condition':'len(value)==HYGO_parameters.N_params'},
            {'parameter':'ninterval','type':int,'parent':'check_type','parent_condition':'value=="interval"','default':1,'specific':'value>0'},
            {'parameter':'global_minima','type':list,'parent':'check_type','parent_condition':'value=="interval"'},
            {'parameter':'neval','type':int,'parent':'check_type','parent_condition':'value=="Neval"'},
            {'parameter':'check_n','type':int,'parent':'check_type','parent_condition':'value=="Relative_change"'},
            {'parameter':'threshold_convergence','type':int,'parent':'check_type','parent_condition':'value=="Relative_change"'},
            {'parameter':'generations_stuck','type':int,'parent':'check_type','parent_condition':'value=="Generation_change"'},
            {'parameter':'mutation_rate','type':float,'parent':'mutation_type','parent_condition':'value=="classic"','specific':'value>0'},
            {'parameter':'LatinN','type':int,'parent':'initialization','parent_condition':'value=="LatinHypercube"','specific':['value>0']},
            {'parameter':'ExploitationType','type':str,'parent':'exploitation','parent_condition':'value','valid_options':['Downhill Simplex','CMA-ES','Scipy']},
            {'parameter':'CMA_Pool','type':str,'parent':'ExploitationType','parent_condition':'value=="CMA-ES"','valid_options':['Population','All']},
            {'parameter':'CMA_gens','type':int,'parent':'ExploitationType','parent_condition':'value=="CMA-ES"','specific':'value>0'},
            {'parameter':'CMA_Lambda','type':int,'parent':'ExploitationType','parent_condition':'value=="CMA-ES"','specific':'value>0'},
            {'parameter':'MaxSimplexCycles','type':int,'parent':'ExploitationType','parent_condition':'value=="Downhill Simplex"','specific':'value>0'},
            {'parameter':'SimplexSize','type':int,'parent':'ExploitationType','parent_condition':'value=="Downhill Simplex"','specific':'value>0'},
            {'parameter':'SimplexPool','type':str,'parent':'ExploitationType','parent_condition':'value=="Downhill Simplex"','valid_options':['Population','All']},
            {'parameter':'SimplexOffspring','type':int,'parent':'ExploitationType','parent_condition':'value=="Downhill Simplex"','specific':'value>0'},
            {'parameter':'SimplexInitialization','type':str,'parent':'ExploitationType','parent_condition':'value=="Downhill Simplex"','valid_options':['BestN','ClosestN'],'default':'BestN'},
            {'parameter':'SimplexCycleChecker','type':int,'parent':'ExploitationType','parent_condition':'value=="Downhill Simplex"','specific':'value>0','default':1},
            {'parameter':'Simplex_R2','type':float,'parent':'ExploitationType','parent_condition':'value=="Downhill Simplex"','specific':'value>0','default':0.999},
            {'parameter':'Simplex_intervals_movement','type':int,'parent':'ExploitationType','parent_condition':'value=="Downhill Simplex"','specific':'value>0','default':300},
            {'parameter':'Simplex_intervals_random_movement','type':int,'parent':'ExploitationType','parent_condition':'value=="Downhill Simplex"','specific':'value>0','default':100},
            {'parameter':'reflection_alpha','type':[int,float],'parent':'ExploitationType','parent_condition':'value=="Downhill Simplex"','specific':'value>0','default':1},
            {'parameter':'expansion_gamma','type':[int,float],'parent':'ExploitationType','parent_condition':'value=="Downhill Simplex"','specific':'value>0','default':2},
            {'parameter':'contraction_rho','type':[int,float],'parent':'ExploitationType','parent_condition':'value=="Downhill Simplex"','specific':'value>0','default':0.5},
            {'parameter':'shrinkage_sigma','type':[int,float],'parent':'ExploitationType','parent_condition':'value=="Downhill Simplex"','specific':'value>0','default':0.5},
            {'parameter':'Scipy_method','type':str,'parent':'ExploitationType','parent_condition':'value=="Scipy"','valid_options':['Minimize'],'default':'Minimize'},
            {'parameter':'Scipy_Initial_Condition','type':str,'parent':'Scipy_method','parent_condition':'value=="Minimize"','valid_options':['Best','Random'],'default':'Best'},
            {'parameter':'Scipy_options','type':dict,'parent':'ExploitationType','parent_condition':'value=="Scipy"','default':{}},
            {'parameter':'Scipy_force_evaluate','type':bool,'parent':'ExploitationType','parent_condition':'value=="Scipy"','default':False},
        ]
    
    
    CONTROL_LAW_PARAMETERS= [
            {'parameter':'operations','type':dict,'parent':'optimization','parent_condition':'value=="Control Law"'},
            {'parameter':'precission','type':int,'parent':'optimization','parent_condition':'value=="Control Law"','specific':'value>0'},
            {'parameter':'control_outputs','type':int,'parent':'optimization','parent_condition':'value=="Control Law"','specific':'value>0'},
            {'parameter':'control_inputs','type':int,'parent':'optimization','parent_condition':'value=="Control Law"','specific':'value>0'},
            {'parameter':'variable_registers','type':int,'parent':'optimization','parent_condition':'value=="Control Law"','specific':'value>0'},
            {'parameter':'constant_resisters','type':int,'parent':'optimization','parent_condition':'value=="Control Law"','specific':'value>0'},
            {'parameter':'number_operations','type':int,'parent':'optimization','parent_condition':'value=="Control Law"','specific':'value>0'},
            {'parameter':'Ninstructions_initial_min','type':int,'parent':'optimization','parent_condition':'value=="Control Law"','specific':'value>0'},
            {'parameter':'Ninstructions_initial_max','type':int,'parent':'optimization','parent_condition':'value=="Control Law"','specific':'value>0'},
            {'parameter':'Minimum_instructions','type':int,'parent':'optimization','parent_condition':'value=="Control Law"','specific':'value>0'},
            {'parameter':'Max_instructions','type':int,'parent':'optimization','parent_condition':'value=="Control Law"','specific':'value>0'},
            {'parameter':'Max_eval_time','type':int,'parent':'optimization','parent_condition':'value=="Control Law"','specific':'value>0'},
            {'parameter':'N_control_points','type':int,'parent':'optimization','parent_condition':'value=="Control Law"','specific':'value>0'},
            {'parameter':'Sensor_range','type':list,'parent':'optimization','parent_condition':'value=="Control Law"'},
            {'parameter':'Control_range','type':list,'parent':'optimization','parent_condition':'value=="Control Law"'},
            {'parameter':'SimplexInterpolation','type':bool,'parent':'optimization','parent_condition':'value=="Control Law"'},
            {'parameter':'reconstruction_neval','type':int,'parent':'optimization','parent_condition':'value=="Control Law"','specific':'value>0'},
            {'parameter':'reconstruction_pop_size','type':int,'parent':'optimization','parent_condition':'value=="Control Law"','specific':'value>0'},
        ]

    def __init__(self,parameters,output_path=None):
        """
        Initialize the HYGO (Genetic Algorithm) object.

        Parameters:
            - parameters: An object containing parameters for the Genetic Algorithm.
            - output_path (str): Path to the output directory. If not provided, the current working directory is used.
        """

        # Set the output path
        if not output_path:
            output_path = os.getcwd() + '/output'
            if not os.path.isdir(output_path):
                os.mkdir(output_path)
        
        # check if the parameters have the initialization method
        
        if hasattr(parameters, 'on_init') and callable(parameters.on_init) and not hasattr(parameters, 'registers'):
            parameters.on_init()
        
        # Initialize HYGO attributes
        self.generation = 0
        self.table = Table()
        self.population = []
        self.parameters = parameters
        self.check_parameters()
        self.output_path = output_path
        self.reached_convergence = False

    def check_parameters(self):
        """
        Check the input parameters for consistency
        """

        # Get the parameter names
        parameter_names = dir(self.parameters)

        # Check compulsory parameters
        for param in self.REQUIRED_PARAMETERS:
            if 'valid_options' in param.keys():
                valid_options = param['valid_options']
            else:
                valid_options = None

            if 'default' in param.keys():
                default = param['default']
            else:
                default = None
            
            if 'specific' in param.keys():
                specific = param['specific']
            else:
                specific = None

            self.parameters = check_parameter(self.parameters, parameter_names, param['parameter'], param['type'], valid_options=valid_options, default=default, specific_condition=specific)
        
        # Check compulsory parameters
        for param in self.CONDITIONAL_PARAMETERS:
            if 'valid_options' in param.keys():
                valid_options = param['valid_options']
            else:
                valid_options = None

            if 'default' in param.keys():
                default = param['default']
            else:
                default = None
            
            if 'specific' in param.keys():
                specific = param['specific']
            else:
                specific = None
            self.parameters = check_conditional_parameter(self.parameters, parameter_names, param['parent'], param['parent_condition'],param['parameter'], param['type'],
                                              valid_options=valid_options, default=default, specific_condition=specific)
        
        # Check compulsory parameters
        if self.parameters.optimization == 'Control Law':
            for param in self.CONTROL_LAW_PARAMETERS:
                if 'valid_options' in param.keys():
                    valid_options = param['valid_options']
                else:
                    valid_options = None

                if 'default' in param.keys():
                    default = param['default']
                else:
                    default = None
                
                if 'specific' in param.keys():
                    specific = param['specific']
                else:
                    specific = None
                self.parameters = check_conditional_parameter(self.parameters, parameter_names, param['parent'], param['parent_condition'],param['parameter'], param['type'],
                                                valid_options=valid_options, default=default, specific_condition=specific)

    def help(self, param_list=None):
        """
        Displays information about the relevant hyperparameters for the optimization

        Parameters:
            - param_list (str/list[str], Optional): List of parameter names to get help. If not introduced all will be displayed
        """
        help_dir = parse_parameter_file(os.path.join(os.path.dirname(__file__), 'tools', 'Parameter_help.txt'))

        if param_list is not None:
            if type(param_list)==str:
                param_list = [param_list]
            elif type(param_list)!=list:
                raise TypeError('The parameter list to get help must be a string or a list of strings')

        for param in self.REQUIRED_PARAMETERS:
            if param_list is None:
                display = True
            else:
                if param["parameter"] in param_list:
                    display = True
                else:
                    display = False
            if display:
                string = f'Parameter: {param["parameter"]}\n'
                string += f'\t Required: True\n'
                string += f'\t Valid types: {param["type"]}\n'

                if 'valid_options' in param.keys():
                    string += f'\t Valid options: {param["valid_options"]}\n'

                if 'default' in param.keys():
                    string += f'\t Default Value: {param["default"]}\n'
                
                if 'specific' in param.keys():
                    if type(param["specific"])==str:
                        string += f'\t Specific Condition: {param["specific"].replace("value",param["parameter"])}\n'
                    else:
                        for i,cond in enumerate(param["specific"]):
                            string += f'\t Specific Condition {i+1}: {cond.replace("value",param["parameter"])}\n'
                if param["parameter"] in help_dir.keys():
                    string += '\t Help: '+help_dir[param["parameter"]][:-2]
                print(string + '\n')
        
        for param in self.CONDITIONAL_PARAMETERS:
            if param_list is None:
                display = True
            else:
                if param["parameter"] in param_list:
                    display = True
                else:
                    display = False
            if display:
                string = f'Parameter: {param["parameter"]}\n'
                string += f'\t Required: False\n'
                string += f'\t Valid types: {param["type"]}\n'
                string += f'\t Dependant on parameter: {param["parent"]}\n'
                if type(param["parent_condition"])==str:
                    string += f'\t Condition for appearance: {param["parent_condition"].replace("value",param["parent"])}\n'
                else:
                    for i,cond in enumerate(param["parent_condition"]):
                        string += f'\t Condition for appearance {i+1}: {cond.replace("value",param["parent"])}\n'

                if 'valid_options' in param.keys():
                    string += f'\t Valid options: {param["valid_options"]}\n'

                if 'default' in param.keys():
                    string += f'\t Default Value: {param["default"]}\n'
                
                if 'specific' in param.keys():
                    if type(param["specific"])==str:
                        string += f'\t Specific Condition: {param["specific"].replace("value",param["parameter"])}\n'
                    else:
                        for i,cond in enumerate(param["specific"]):
                            string += f'\t Specific Condition {i+1}: {cond.replace("value",param["parameter"])}\n'
                if param["parameter"] in help_dir.keys():
                    string += '\t Help: '+help_dir[param["parameter"]][:-2]
                print(string + '\n')

        for param in self.CONTROL_LAW_PARAMETERS:
            if param_list is None:
                display = True
            else:
                if param["parameter"] in param_list:
                    display = True
                else:
                    display = False
            if display:
                string = f'Parameter: {param["parameter"]}\n'
                string += f'\t Required: False\n'
                string += f'\t Valid types: {param["type"]}\n'
                string += f'\t Dependant on parameter: {param["parent"]}\n'
                if type(param["parent_condition"])==str:
                    string += f'\t Condition for appearance: {param["parent_condition"].replace("value",param["parent"])}\n'
                else:
                    for i,cond in enumerate(param["parent_condition"]):
                        string += f'\t Condition for appearance {i+1}: {cond.replace("value",param["parent"])}\n'

                if 'valid_options' in param.keys():
                    string += f'\t Valid options: {param["valid_options"]}\n'

                if 'default' in param.keys():
                    string += f'\t Default Value: {param["default"]}\n'
                
                if 'specific' in param.keys():
                    if type(param["specific"])==str:
                        string += f'\t Specific Condition: {param["specific"].replace("value",param["parameter"])}\n'
                    else:
                        for i,cond in enumerate(param["specific"]):
                            string += f'\t Specific Condition {i+1}: {cond.replace("value",param["parameter"])}\n'
                if param["parameter"] in help_dir.keys():
                    string += '\t Help: '+help_dir[param["parameter"]][:-2]
                print(string + '\n')

    def generate_population(self):
        """
        Generate the initial population for the genetic algorithm.
        """

        # Add a Population object to the population list
        if self.population == []:
            if type(self.parameters.pop_size)==list:
                pop = Population(self.parameters.pop_size[0],1)
            else:
                pop = Population(self.parameters.pop_size,1)
        else:
            raise Exception('First population already created')

        # Generate the population
        pop.generate_pop(self.parameters,self.table)

        # Save the data in the population attribute
        self.population.append(pop)
        self.generation+=1

    def evaluate_population(self, ngen=None):
        """
        Evaluate the fitness of individuals in the population.

        Parameters:
            - ngen (int, Optional): Generation number to evaluate. If not provided, evaluate the latest generation.

        Returns:
            - checker (bool): Flag indicating that the maximum number of evaluations is reached or not.
        """

        # If the number of generations to be evaluated is not introduced, the last one will be evaluated
        if not ngen:
            ngen = len(self.population)-1

        # Checj that the generation number is well introduced
        if ngen>(len(self.population)-1):
            raise ValueError('The introduced generation value does not exist')
        
        # Obtain the population object
        pop = self.population[ngen]

        # Backup HYGO object if required
        if self.parameters.security_backup:
            import dill
            file = open(self.output_path+'/HYGO_backup.obj','wb')
            dill.dump(self,file)
            file.close()

        if not len(pop.idx_to_evaluate):
            # Determine individuals to be evaluated
            opes = pop.data['Operation','Type'].values.tolist()
            idx_to_evaluate = []

            # Exclude the individuals from elitism and replication from evaluation
            for i,op in enumerate(opes):
                if op!='Elitism' and op!='Replication':
                    idx_to_evaluate.append(i)

            idx_to_evaluate = [idx_to_evaluate.copy() for _ in range(self.parameters.repetitions)]
            # Save the individuals to be evaluated
            self.population[ngen].idx_to_evaluate = copy.deepcopy(idx_to_evaluate)
        else:
            idx_to_evaluate = copy.deepcopy(pop.idx_to_evaluate)
        
        # Perform population evaluation
        self.table,checker = self.population[ngen].evaluate_population(idx_to_evaluate,self.parameters,self.table,self.output_path)
        self.population[ngen].sort_pop()

        return checker

    def evolve_population(self):
        """
        Evolve the population to the next generation.
        """

        # Check that the last generation is evaluated
        if self.population[-1].state != 'Evaluated':
            raise Exception('Generation '+str(len(self.population))+' not evaluated yet')

        # Obtain the population size of the new population
        if type(self.parameters.pop_size)==list:
            new_size = self.parameters.pop_size[len(self.population)]
        else:
            new_size = self.parameters.pop_size

        # Create the new population object with the last generation
        new_pop = self.population[-1].evolve_pop(self.parameters,self.table,new_size)

        # Save the new population
        self.population.append(new_pop)
        self.generation+=1

    def go(self,ngen=None):
        """
        Run the genetic algorithm for a specified number of generations.

        Parameters:
            - ngen (int): Number of generations to run. If not provided, use the value specified in parameters.

        """
        # If the number of generations is not introduced, use the plant parameters
        if not ngen:
            ngen = self.parameters.ngen

        if self.parameters.verbose:
            print('**************Running '+self.parameters.name+' problem**************\n')

        # Inialize the checker flag
        checker=True

        # Create and evaluate the first population (if already created just evaluate it)
        if self.population==[]:
            '''if not self.parameters.verbose:
                print('Gen 1/'+str(self.parameters.ngen))'''
            
            # Generate the first population
            self.generate_population()
            
            # Evaluate the population
            checker = self.evaluate_population()
            
        elif self.population[0].state == 'Generated':
            # Evaluate the population
            checker = self.evaluate_population()
        
        # Initialize the convergence as false
        convergence=False

        # Perform the Downhill simplex exploitation if selected
        checker, convergence = self.exploitation(0,checker,convergence)
        
        # Continue with the optimisation until the maximum number of generations, 
        #   maximum number of evaluations or convergence
        while self.generation<ngen and not convergence and checker:
            
            '''if not self.parameters.verbose:
                print('Gen '+str(self.generation+1)+'/'+str(self.parameters.ngen))'''

            # Check that the last population was evaluated
            if self.population[-1].state=='Evaluated':
                # Perform the exploitation if necessary
                checker, convergence = self.exploitation(-1,checker,convergence)
            else:
                # Evaluate the population
                checker = self.evaluate_population()
                
                # Perform the exploitation
                checker, convergence = self.exploitation(-1,checker,convergence)
            
            # Check convergence    
            if self.parameters.check_convergence and not convergence:
                convergence = self.check_convergence()
            
            # Evolve the previous generation
            self.evolve_population()
            
            # Evaluate the new population
            checker = self.evaluate_population()
            
            # Perform the exploitation
            checker, convergence = self.exploitation(-1,checker,convergence)

        self.reached_convergence=convergence

        if self.parameters.verbose:
            print('---FINISHED---')
        
        if self.parameters.name!='ReconstructionProblem':
            # Show the optimisation rsults
            self.display_convergence()

    def exploitation(self,gen,checker,convergence):
        '''
        Performs the exploitation while checking which steps are required.
        
        Parameters:
            - gen (int): index of the population to be exploited.
            - checker (bool): A flag indicating whether the convergence by the number of individuals is reached.
            - convergence (bool): Flag indicating that convergence is reached.
        
        Returns:
            - gen (int): index of the population to be exploited.
            - checker (bool): A flag indicating whether the convergence by the number of individuals is reached.
        '''
        #Check that the exploitation was done and finished
        if hasattr(self.parameters,'exploitation') and self.parameters.exploitation and checker:
            # Perform the Downhill simplex exploitation if selected
            if self.parameters.ExploitationType == 'Downhill Simplex':
                
                # Check that the simplex was initialized
                if not hasattr(self.population[gen],'simplex_state'):
                    if self.parameters.verbose:
                        print('################ Exploitation for generation '+str(self.population[gen].generation)+' ################')
                        
                    self.population[gen].initialize_simplex(self.parameters,self.table)
                    self.table,checker,convergence = self.population[gen].exploitation_simplex(self.parameters,self.table,self.output_path)
                # Check that the simplex was finised
                elif not self.population[gen].simplex_state=='Simplex done' and self.population[gen].simplex_state=='Exploitation initialized':
                    if self.parameters.verbose:
                        print('################ Exploitation for generation '+str(self.population[gen].generation)+' ################')
                    
                    self.table,checker,convergence = self.population[gen].exploitation_simplex(self.parameters,self.table,self.output_path)
            elif self.parameters.ExploitationType == 'CMA-ES':
                # Compute the exploitation
                # Check that the simplex was initialized
                if not hasattr(self.population[gen],'cma_es_state'):
                    if self.parameters.verbose:
                        print('################ Exploitation for generation '+str(self.population[gen].generation)+' ################')
                    self.population[gen].cma_es_state = None
                    self.table,checker,convergence = self.population[gen].CMA_ES_exploitation(self.parameters,self.table,self.output_path)
                elif self.population[gen].cma_es_state != 'CMA-ES done':
                    self.table,checker,convergence = self.population[gen].CMA_ES_exploitation(self.parameters,self.table,self.output_path)
            elif self.parameters.ExploitationType == 'Scipy':
                # Compute the exploitation
                if not hasattr(self.population[gen],'cma_es_state'):
                    if self.parameters.verbose:
                        print('################ Exploitation for generation '+str(self.population[gen].generation)+' ################')
                    self.population[gen].scipy_state = None
                    self.table,checker,convergence = self.population[gen].Scipy_exploitation(self.parameters,self.table,self.output_path)
                elif self.population[gen].scipy_state != 'Scipy Done':
                    self.table,checker,convergence = self.population[gen].Scipy_exploitation(self.parameters,self.table,self.output_path)
            else:
                raise ValueError('Only Downhill Simplex, CMA-ES and Scipy are available as explotation methods')
        
        return checker, convergence

    def display_convergence(self):
        '''
        Shows the results of the ooptimisation
        '''
        best_idx = int(self.population[-1].data.iloc[0,0])
        best_cost = self.table.individuals[best_idx].cost
        # best_idx,best_cost = self.table.give_best(1)
        print('Best cost: ',best_cost)
        print('Best individual: ',int(best_idx))
        print('\nBest individual parameters:\n')
        
        best_params = self.table.individuals[int(best_idx)].parameters
        
        if self.parameters.optimization == 'Parametric':
            for i in range(self.parameters.N_params):
                print('\tb'+str(i)+' = '+str(best_params[i]))
                
        elif self.parameters.optimization == 'Control Law':
            for i in range(self.parameters.control_outputs):
                print('\tb'+str(i)+' = '+str(best_params[i]))

    def check_convergence(self):
        
        """
        Check for Convergence

        This method checks for convergence based on the specified criteria in the parameters.
        The convergence criteria include interval-based checks, relative change, and generation change.

        Returns:
            bool: True if convergence criteria are met, indicating the algorithm should stop.
                  False otherwise.

        Convergence Criteria:
            - Interval-based: Checks if individuals' parameters are within a specified interval
                              around predefined global minima.
            - Relative Change: Checks if the relative change in the cost function is below a threshold.
            - Generation Change: Checks if the costs remain constant over a specified number of generations.

        Note:
            The specific convergence criteria are determined by the parameters provided to the genetic algorithm.

        """

        # Obtain the intervals for convergence for each parameter
        if self.parameters.check_type=='interval':
            interval = []
            for i in range(self.parameters.N_params):
                if type(self.parameters.Nb_bits)!=int:
                    if len(self.parameters.Nb_bits)>1:
                        interval.append((self.parameters.params_range[i][1]-self.parameters.params_range[i][0])/(2**self.parameters.Nb_bits[i]-1))
                    else:
                        interval.append((self.parameters.params_range[i][1]-self.parameters.params_range[i][0])/(2**self.parameters.Nb_bits[0]-1))
                else:
                    interval.append((self.parameters.params_range[i][1]-self.parameters.params_range[i][0])/(2**self.parameters.Nb_bits-1))
        
        # Check for relative change convergence
        elif self.parameters.check_type=='Relative_change':
            if len(self.table.individuals)>self.parameters.check_n:
                costs = np.array(self.table.costs[-self.parameters.check_n:])
                
                # Check if the relative change in costs is below the threshold
                if (np.max(costs)-np.min(costs))/np.min(costs)<(self.parameters.threshold_convergence/100):
                    return True
                else:
                    return False
        
        # Check for generation change convergence
        elif self.parameters.check_type=='Generation_change':
            if self.generation>=self.parameters.generations_stuck:
                costs = []
                for pop in self.population:
                    costs.append(pop.data.loc[0,'Costs'])
                real_costs = costs[-self.parameters.generations_stuck:]
                
                # Check if costs remain constant over a specified number of generations
                if len(np.unique(real_costs))==1:
                    return True
                else:
                    return False
        else:
            return False

        # Loop to check the individuals
        for ind in self.table.individuals:
            params = ind.parameters
            
            # Check for interval-based convergence 
            if self.parameters.check_type=='interval':
                convergence = []
                
                # Obtain the global minima
                global_minima = np.array(self.parameters.global_minima)
                
                # Check paramiter by parameter if inside the threshold
                if len(global_minima.shape)>1:
                    for minima in global_minima:
                        convergence = []
                        for i in range(len(params)):
                            if (params[i]<(minima[i]+self.parameters.ninterval*interval[i])) and (params[i]>(minima[i]-self.parameters.ninterval*interval[i])):
                                convergence.append(True)
                            else:
                                convergence.append(False)
                        if (len(convergence)-sum(convergence))==0:
                            print('---->Early convergence')
                            return True
                else:
                    for i in range(len(params)):
                        if (params[i]<(self.parameters.global_minima[i]+self.parameters.ninterval*interval[i])) and (params[i]>(self.parameters.global_minima[i]-self.parameters.ninterval*interval[i])):
                            convergence.append(True)
                        else:
                            convergence.append(False)
                    if (len(convergence)-sum(convergence))==0:
                        print('---->Early convergence')
                        return True
        
        return False

    def save(self,path=None, specific_save=False):
        
        """
        Save the State of the Genetic Algorithm

        Args:
            path (str, optional): The directory path where the data will be saved. If not provided, the output_path
                                attribute will be used. If neither is provided, a ValueError will be raised.
            specific_save (bool, optional): If True, the method will save individual files with specific information
                                        instead of a single object. Default is False.

        Raises:
            ValueError: If no path is provided and neither output_path nor path is available.

        Note:
            The method can save either the entire HYGO object or specific information in separate files.

        """
        # Check if a path is provided, if not, use output_path or raise an error
        if not path and not self.output_path:
            raise ValueError('No available path to save the information')
        if not path:
            path = self.output_path
        
        # Use dill for general object serialization or save specific information in separate files
        if not specific_save:
            import dill
            # Save the entire HYGO object using dill
            file = open(path+'HYGO.obj','wb')
            dill.dump(self,file)
        else:
            # Save specific information in separate files
            attributes = [attribute for attribute in dir(self.parameters)
                if not attribute.startswith('__')
                and not callable(getattr(self.parameters, attribute))]
            
            # Save general information in a text file   
            file = open(path+'/General.txt','w')
            file.write('Generation='+str(self.generation)+'\n')
            if self.output_path:
                file.write('Path='+str(path)+'\n')
            for atr in attributes:
                file.write(atr+'='+str(getattr(self.parameters,atr))+'\n')
            file.close()

            # Save information about the table in a text file
            file = open(path+'/Table.txt','w')
            file.write('Latest Indiv='+str(self.table.latest_indiv)+'\n')
            file.write('Hashlist='+str(self.table.hashlist)+'\n')
            file.write('Costs='+str(self.table.costs)+'\n')
            for i,ind in enumerate(self.table.individuals):
                file.write('---Individual '+str(i)+'---\n')
                file.write('Chromosome='+str(ind.chromosome)+'\n')
                file.write('Cost='+str(ind.cost)+'\n')
                file.write('Parameters='+str(ind.parameters)+'\n')
                file.write('Path='+str(ind.path)+'\n')
                file.write('Occurrences='+str(ind.ocurrences)+'\n')
                file.write('Hash='+str(ind.hash)+'\n')

            # Save information about populations in a text file
            file = open(path+'/Populations.txt','w')
            for pop in self.population:
                file.write('---Population '+str(pop.generation)+' ---\n')
                file.write('generation='+str(pop.generation)+'\n')
                file.write('Nind='+str(pop.Nind)+'\n')
                file.write('repetition='+str(pop.repetition)+'\n')
                file.write('idx_to_evaluate='+str(pop.idx_to_evaluate)+'\n')
                file.write('idx_to_check='+str(pop.idx_to_check)+'\n')
                file.write('state='+str(pop.state)+'\n')
                if self.parameters.exploitation and hasattr(pop,'simplex_idx'):
                    file.write('simplex_idx='+str(pop.simplex_idx)+'\n')
                    file.write('simplex_costs='+str(pop.simplex_costs)+'\n')
                    if hasattr(pop,'simplex_centroid'):
                        file.write('simplex_centroid='+str(pop.simplex_centroid)+'\n')
                    file.write('simplex_cycle='+str(pop.simplex_cycle)+'\n')
                    file.write('simplex_state='+str(pop.simplex_state)+'\n')
                    file.write('simplex_memory='+str(pop.simplex_memory)+'\n')
                
                # Save population data to a CSV file
                pop.data.to_csv(path+'/Populations'+str(pop.generation)+'.csv',sep=',',header=False,index=False)

            file.close()

    @classmethod
    def load_security(cls,path):
        """
        Load HYGO Object from Security Backup

        Args:
            path (str): The directory path where the security backup is stored.

        Returns:
            HYGO: An instance of the HYGO class loaded from the security backup.

        Note:
            The method loads the HYGO object from a security backup created during an interrupted run.

        """

        import dill
        
        # Load the HYGO object from the security backup using dill
        obj = cls.load(path+'/HYGO_backup.obj')
        print('-->Done')

        '''del obj.population[-1]

        obj.generation = 2
        obj.parameters.tournament_size = int(0.07*obj.parameters.pop_size)'''
        # Get the interrupted generation from the loaded HYGO object
        interrupted_gen = obj.generation

        print(f'-----Loading data from interrupted generation {interrupted_gen}-----')

        # Adjust the path for the interrupted generation
        path = path + '/Gen'+str(int(interrupted_gen))

        # Load the population and table backups from individual files
        file = open(path+'/pop_backup.obj','rb')
        pop = dill.load(file)
        file.close()
        file = open(path+'/table_backup.obj','rb')
        table = dill.load(file)
        file.close()

        # Update the HYGO object with the loaded table and population backups
        obj.table = table
        obj.population[-1] = pop

        print('-->Done')

        return obj

    @classmethod
    def load(cls,path, specific_save=False):
        """
        Load HYGO Object from a Saved State

        Args:
            path (str): The directory path where the saved state is stored.
            specific_save (bool): If True, load specific files from the saved state.

        Returns:
            HYGO: An instance of the HYGO class loaded from the saved state.

        Note:
            The method loads the HYGO object from a saved state, including general data, table data,
            and populations data.

        """
        
        print('-----Loading data-----\n')

        if not specific_save:
            # If loading the entire saved state
            import dill
            
            # Load the HYGO object using dill
            file = open(path,'rb')
            return dill.load(file)
        else:
            # If loading specific files from the saved state
            import pandas as pd

            def has_numbers(inputString):
                return any(char.isdigit() for char in inputString)

            def cost_function(parameters,HYGO_params,path):
                return 0,0
            
            from .tools.DummyParams import DummyParameters
            params = DummyParameters()

            print('Loading General data')
            with open(path+'/General.txt') as f:
                # Read and parse the general information from the file
                gen = f.readline()
                gen = int(gen.split('=')[1][:-1])
                path = f.readline()
                path = path.split('=')[1][:-1]
                parameters = {}
                for line in f:
                    # Remove any whitespace and skip blank or commented lines
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    # Split the line into key and value
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        try:
                            # Try to evaluate the value as a Python literal
                            parameters[key] = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            # Fallback: keep it as a string if evaluation fails
                            parameters[key] = value
                for key in parameters.keys():
                    setattr(params,key,parameters[key])

                f.close()

                # Set the cost function attribute in params
                params.cost_function = lambda parameters,HYGO_params,path=None : cost_function(parameters,HYGO_params,path)
                params.plotter = None
                
                # Create an instance of HYGO class with DummyParameters and set the generation
                obj = cls(params,path)
                obj.generation  = gen

            print('--->DONE\n')

            # Load the table data
            table = Table()

            print('Loading table data')
            with open(path+'/Table.txt') as f:
                line = f.readline()
                table.latest_indiv = int(line.split('=')[1][:-1])
                line = f.readline()
                line=line.split('=')[1][1:-2]
                line=line.split(',')
                hash=[]
                for val in line:
                    hash.append(int(val))
                table.hashlist = hash
                line = f.readline()
                line=line.split('=')[1]
                line=line[1:-2]
                line=line.split(',')
                cost=[]
                for val in line:
                    cost.append(float(val))
                table.costs = cost
                for _ in range(table.latest_indiv):
                    ind = Individual()
                    f.readline()
                    line=f.readline()
                    line=line.split('=')[1]

                    if ',' not in line:
                        if ']' not in line:
                            line=line[1:-1]
                            line=line.split(' ')
                            line2=f.readline()[:-2]
                            line2=line2.split(' ')
                            while line2[0]=='':
                                line2.pop(0)
                            line=line+line2
                        else:
                            line=line[1:-2]
                            line=line.split(' ')
                    else:
                        line=line[1:-2]
                        line=line.split(',')

                    ind.chromosome = [int(num) for num in line]

                    line=f.readline()
                    line=line.split('=')[1][:-1]
                    if line=='None':
                        ind.cost = params.badvalue
                    else:
                        ind.cost = float(line)

                    line=f.readline()
                    line=line.split('=')[1][1:-2]
                    line=line.split(',')
                    ind.parameters = [float(val) for val in line]

                    line=f.readline()
                    line=line.split('=')[1][1:-2]
                    ind.path=line.replace("'",'').replace('//','/').split(',')

                    line=f.readline()
                    line=line.split('=')[1][:-1]
                    ind.ocurrences = int(line)

                    line=f.readline()
                    line=line.split('=')[1][:-1]
                    ind.hash = int(line)

                    table.individuals.append(ind)
                f.close()

            obj.table=table
            print('--->DONE\n')

            # Load the population data
            print('Loading populations data')
            for i in range(gen):
                print('\t-Loading population '+str(i+1))
                with open(path+'/Populations.txt') as f:
                    f.readline()
                    line=f.readline()
                    generation = int(line.split('=')[1][:-1])
                    line=f.readline()
                    Nind = int(line.split('=')[1][:-1])

                    pop=Population(Nind,generation)

                    line=f.readline()
                    pop.repetition = int(line.split('=')[1][:-1])
                    line=f.readline()
                    pop.state = line.split('=')[1][:-1]

                    if params.exploitation:
                        line=f.readline()
                        line=line.split('=')[1][1:-2]
                        line=line.split(',')
                        pop.simplex_idx = [int(float(val)) for val in line]

                        line=f.readline()
                        line=line.split('=')[1][1:-2]
                        line=line.split(',')
                        pop.simplex_costs = [float(val) for val in line]

                        line=f.readline()
                        line=line.split('=')[1][1:-2]
                        line=line.split(',')
                        pop.simplex_centroid = [float(val) for val in line]

                        line=f.readline()
                        pop.simplex_cycle = int(line.split('=')[1][:-1])

                    f.close()

                header = [("Individuals",''),
                            ("Costs",''),
                            ("Uncertainty",'Minimum'), 
                            ("Uncertainty",'All'), 
                            ("Parents", "first"), 
                            ("Parents", "second"),
                            ('Operation','Type'),
                            ('Operation','Point_parts')]
                
                for j in range(pop.repetition):
                    header = header + [("Rep "+ str(j+1),'Evaluation_time'),
                                        ("Rep "+ str(j+1),'Path'),
                                        ("Rep "+ str(j+1),'Cost'),
                                        ("Rep "+ str(j+1),'Cost_terms')]
                pop.data = pd.read_csv(path+'/Populations'+str(i+1)+'.csv',sep=',',header=None,names=header)
                obj.population.append(pop)
            print('--->DONE\n')

            return obj

    def convergence(self, fitness=False, gens=None, individual=None, save=None, show=True):
        """
        Plot convergence information for a genetic algorithm.

        Parameters:
            fitness (bool): If True, plots 1/cost instead of cost.
            gens(list): list of generations to plot
            individual(int): individual idx to build the ancestry graph
            save (str): If specified, path to save the figure.
            show (bool): If the images are to be displayed.

        Returns:
            None
        """
        if not individual:
            individual,_ = self.table.give_best(1)
            individual = int(individual[0])

        if save:
            self.plot_J_evolution(fitness,save=save+'J_evolution.png')
            self.plot_operations(save=save+'Operations_distribution.png')
            self.plot_ancestry_graph(gens,save=save+'Ancestry_graph.png')
            self.plot_individual_ancestry(individual=individual, save=save+f'Individual_{individual}_Ancestry_graph.png')
            if self.parameters.optimization == 'Parametric':
                self.plot_parameter_cost_correlation(save=save+'Correlation')
                self.plot_parameter_diversity(save=save+'Parameter_diversity.png')
                self.plot_partial_dependence(save=save+'Parameter_dependence.png')
                self.plot_dimensionality_reduction(save=save+'Dimensionality_reduction.png')
                
        else:
            self.plot_J_evolution(fitness)
            self.plot_operations()
            self.plot_ancestry_graph(gens)
            self.plot_individual_ancestry(individual)
            if self.parameters.optimization == 'Parametric':
                self.plot_parameter_cost_correlation()
                self.plot_parameter_diversity()
                self.plot_partial_dependence()
                self.plot_dimensionality_reduction()
                

        if hasattr(self.parameters, 'plotter') and callable(self.parameters.plotter):
            self.parameters.plotter(self)

        if show:
            plt.show()

    def plot_dimensionality_reduction(self, save=None):
        """
        Plots a 2x3 grid of dimensionality reduction projections (PCA, MDS, Isomap, LLE, t-SNE, UMAP) 
        for the individuals in the population, colored by their associated cost.

        This helps visualize the structure and diversity of individuals in parameter space, 
        as well as potential clusters or convergence behavior during optimization.

        Parameters:
            save (str): Optional. If provided, path to save the resulting figure.

        Returns:
            None
        """

        try:
            importlib.import_module('sklearn')
            
        except ImportError:
            print("sklearn package is required for the build_ancestry_graph function")
            return
        try:
            importlib.import_module('umap')
            
        except ImportError:
            print("umap package is required for the build_ancestry_graph function")
            return
        from sklearn.decomposition import PCA
        from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, TSNE
        import umap

        # Dimensionality reduction techniques
        reducers = {
            "PCA": PCA(n_components=2),
            "MDS": MDS(n_components=2, dissimilarity="euclidean", random_state=42),
            "Isomap": Isomap(n_components=2),
            "LLE": LocallyLinearEmbedding(n_components=2, method='standard'),
            "t-SNE": TSNE(n_components=2, random_state=42, perplexity=30),
            "UMAP": umap.UMAP(n_components=2, random_state=42)
        }

        parameters = []
        costs = []
        for ind in self.table.individuals:
            parameters.append(ind.parameters)
            costs.append(ind.cost)
        parameters = np.asarray(parameters)
        costs = np.asarray(costs)

        idx = np.where(costs>0.99*self.parameters.badvalue)

        parameters = np.delete(parameters,idx,axis=0)
        costs = np.delete(costs,idx)

        X_scaled = (parameters - np.min(parameters,axis=0))/(np.max(parameters,axis=0) - np.min(parameters,axis=0))

        # Apply each reducer
        embeddings = {}
        for name, reducer in reducers.items():
            embeddings[name] = reducer.fit_transform(X_scaled)

        # Plotting 2x3 grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        axes = axes.flatten()

        for ax, (name, embedding) in zip(axes, embeddings.items()):
            sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=costs, cmap='viridis', s=20, alpha=0.7)
            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.colorbar(sc, ax=axes, orientation='horizontal', fraction=0.03, pad=-0.1, label='Cost', location = 'top')
        plt.suptitle("Dimensionality Reduction of Genetic Optimization Individuals", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save:
            plt.savefig(save, bbox_inches='tight')

    def build_ancestry_graph(self, gens=None):
        """
        Builds a directed ancestry graph of individuals across generations, where nodes represent individuals 
        and edges point from parents to offspring.

        The graph is built using parent indices stored in each generation's population data, supporting both 
        intra-generation and inter-generation parent references. Optionally, a subset of generations can be specified.

        Parameters:
            gens (list or range, optional): Specific generations to include. If None, all generations are used.

        Returns:
            G (networkx.DiGraph): Directed graph with nodes as (generation, individual_id) tuples.
            J (dict): Dictionary mapping individual_id to its cost for reference or coloring.
        """
        try:
            importlib.import_module('networkx')
            
        except ImportError:
            print("Networkx package is required for the build_ancestry_graph function")
            return

        import networkx as nx
        G = nx.DiGraph()
        J = {}

        if not gens:
            gens = range(len(self.population))

        for gen_idx in gens:
            pop = self.population[gen_idx]
            parents_first = pop.data[('Parents', 'first')]
            parents_second = pop.data[('Parents', 'second')]
            idx_individuals = pop.data['Individuals']
            costs = pop.data['Costs']

            for ind_idx in range(pop.data.shape[0]):
                node_id = (gen_idx, int(idx_individuals[ind_idx]))
                G.add_node(node_id)
                J[int(idx_individuals[ind_idx])] = costs[ind_idx]

                if gen_idx > 0:
                    if parents_first[ind_idx] in self.population[gen_idx-1].data['Individuals'].values:
                        parent1 = (gen_idx - 1, int(parents_first[ind_idx]))
                        G.add_edge(parent1, node_id)
                    if parents_first[ind_idx] in self.population[gen_idx].data['Individuals'].values:
                        parent1 = (gen_idx, int(parents_first[ind_idx]))
                        G.add_edge(parent1, node_id)

                    if parents_second[ind_idx]>-1:
                        if parents_second[ind_idx] in self.population[gen_idx-1].data['Individuals'].values:
                            parent2 = (gen_idx - 1, int(parents_second[ind_idx]))
                            G.add_edge(parent2, node_id)
                        if parents_second[ind_idx] in self.population[gen_idx].data['Individuals'].values:
                            parent2 = (gen_idx, int(parents_second[ind_idx]))
                            G.add_edge(parent2, node_id)

        return G, J

    def plot_ancestry_graph(self, gens=None, save=None):
        """
        Plots the full ancestry graph across generations, with nodes colored by individual cost.

        The x-axis represents generations (left to right), and the y-axis vertically stacks individuals within 
        each generation. Nodes are colored using a cost-based colormap, with invalid or placeholder costs 
        masked and adjusted for visualization.

        Parameters:
            gens (list or range, optional): Specific generations to include. If None, all generations are plotted.
            save (str, optional): Path to save the resulting figure. If not specified, the plot is shown interactively.

        Returns:
            None
        """
        try:
            importlib.import_module('networkx')
        except ImportError:
            print("Networkx package is required for the build_ancestry_graph function")
            return
        import networkx as nx

        G, J = self.build_ancestry_graph(gens)

        # Organize nodes by generation
        gen_dict = {}
        for node in G.nodes:
            gen, idx = node
            if gen not in gen_dict:
                gen_dict[gen] = []
            gen_dict[gen].append(idx)

        # Sort generations and compute positions
        pos = {}
        labels = {}
        costs = []

        for gen in sorted(gen_dict.keys()):
            ids = sorted(gen_dict[gen])
            for i, ind in enumerate(ids):
                node_id = (gen,ind)
                pos[node_id] = (gen, -i)  # x = generation, y = vertical position
                labels[node_id] = str(int(ind))
                costs.append(J[int(ind)])

        costs = np.asarray(costs)
        idx = np.where(costs>0.99*self.parameters.badvalue)
        costs[idx] = -self.parameters.badvalue
        max_val = np.max(costs)
        costs[idx] = max_val

        plt.figure(figsize=(18, 8))
        nx.draw(G, pos, labels=labels, with_labels=True, node_size=250, node_color=costs, cmap=plt.cm.coolwarm, arrows=True, arrowsize=7)
        plt.title("Ancestry Tree (Generations Left to Right)")
        plt.xlabel("Generation")
        plt.ylabel("Individual Index (Inverted)")
        plt.tight_layout()

        if save:
            plt.savefig(save, bbox_inches='tight')

    def plot_individual_ancestry(self, individual=None, save=None):
        """
        Plots the ancestry tree of a specific individual, tracing all of its parents recursively across generations.

        Each node represents an ancestor (including the target individual), and is colored by the genetic operation 
        that produced it (e.g., Crossover, Mutation). The individual appears at the rightmost generation, with 
        edges pointing back in time. A legend maps colors to operation types.

        Parameters:
            individual (int): The individual index to trace ancestry for. The most recent generation is used if multiple found.
            save (str, optional): Path to save the resulting figure. If None, the plot is displayed interactively.

        Returns:
            None
        """

        try:
            importlib.import_module('networkx')
        except ImportError:
            print("Networkx package is required for the build_ancestry_graph function")
            return
        from matplotlib.patches import Patch
        import networkx as nx

        G, J = self.build_ancestry_graph()

        nodes = np.asarray(G.nodes())
        
        def collect_ancestry_subgraph(G, node, visited):
            if node in visited:
                return
            visited.add(node)
            for parent in G.predecessors(node):
                collect_ancestry_subgraph(G, parent, visited)

        visited_nodes = set()

        #find individual
        idx = np.where(nodes[:,1] == individual)[0]
        gens = nodes[idx,0]
        gen = int(gens[np.argmax(gens)])

        collect_ancestry_subgraph(G, (gen,individual), visited_nodes)
        subgraph = G.subgraph(visited_nodes)

        if hasattr(self.parameters,'ExploitationType'):
            exploitation = self.parameters.ExploitationType
        else:
            exploitation = ''
        colors = {
                'Random': 'grey', 'Elitism': 'red', 'Replication': 'lightcoral',
                'Mutation': 'indianred', 'Crossover': 'darkred'
            }
        if exploitation == 'Downhill Simplex':
            colors.update({
                'Reflection': 'royalblue', 'Contraction': 'lightsteelblue',
                'Shrinked': 'navy', 'Expanded': 'skyblue',
                'Correction': 'blueviolet', 'CorrectionRandom': 'violet'
            })
        elif exploitation == 'CMA-ES':
            colors['CMA-ES'] = 'royalblue'
        elif exploitation == 'Scipy':
            colors['Scipy'] = 'royalblue'

        pos, gen_to_y = {}, {}
        for gen, idx in sorted(subgraph.nodes):
            if gen not in gen_to_y:
                gen_to_y[gen] = 0
            pos[(gen, idx)] = (gen, -gen_to_y[gen])
            gen_to_y[gen] += 1
        
        operation_colors = []
        for node in subgraph.nodes:
            gen, idx = node
            inds_col = self.population[gen].data['Individuals']
            ops_col = self.population[gen].data[('Operation', 'Type')]

            idx_match = np.where(inds_col.values == idx)[0]
            if len(idx_match) > 0:
                op_type = ops_col.iloc[idx_match[0]]
                operation_colors.append(colors.get(op_type, 'gray'))
            else:
                operation_colors.append('gray')

        labels = {node: str(node[1]) for node in subgraph.nodes}

        legend_elements = [Patch(facecolor=clr, edgecolor='k', label=op) for op, clr in colors.items()]

        plt.figure(figsize=(12, 6))
        nx.draw(subgraph, pos, with_labels=True, labels=labels,
                node_size=400, node_color=operation_colors, arrows=True, arrowsize=12)
        plt.title(f"Ancestry of Individual {individual}")
        plt.xlabel("Generation →")
        plt.ylabel("Ancestor Stack")
        plt.legend(handles=legend_elements, title="Operation", loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        plt.tight_layout()
        if save:
            plt.savefig(save, bbox_inches='tight')

    def plot_parameter_cost_correlation(self, method='pearson', save=None):
        """
        Plots correlation heatmaps between optimization parameters and Cost.

        Parameters:
            threshold (float): Cost threshold above which individuals are excluded. Defaults to self.parameters.badvalue / 1000.
            method (str): Correlation method: 'pearson', 'spearman', or 'kendall'.

        Returns:
            None
        """
        try:
            importlib.import_module('seaborn')
            
        except ImportError:
            print("seaborn package is required for the build_ancestry_graph function")
            return
        import seaborn as sns
        parameters = []
        costs = []
        for ind in self.table.individuals:
            parameters.append(ind.parameters)
            costs.append(ind.cost)
        parameters = np.asarray(parameters)
        costs = np.asarray(costs)

        n_gens = len(self.population)
        if n_gens<4:
            n_cols=n_gens
        else:
            n_cols = 4
        n_rows = int(np.ceil(n_gens / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        all_data = []

        for gen_idx, pop in enumerate(self.population):
            idx = pop.data['Individuals'].values.astype(int)
            costs_pop = costs[idx]
            params_pop = parameters[idx, :]

            # Filter by threshold
            valid_mask = costs_pop < self.parameters.badvalue
            if not np.any(valid_mask):
                axes[gen_idx].axis('off')
                continue

            params_valid = params_pop[valid_mask]
            costs_valid = costs_pop[valid_mask]

            df = pd.DataFrame(params_valid, columns=[f"Param_{i}" for i in range(params_valid.shape[1])])
            df['Costs'] = costs_valid
            all_data.append(df)

            corr = df.corr(method=method)
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[gen_idx], cbar=False, vmin=-1, vmax=1)
            axes[gen_idx].set_title(f"Generation {gen_idx + 1}")

        # Hide unused subplots
        for i in range(n_gens, len(axes)):
            axes[i].axis('off')

        fig.suptitle("Per-Generation Parameter-Cost Correlation", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        if save:
            fig.savefig(save+'_gens.png', bbox_inches='tight')

        # Full population heatmap
        if all_data:
            df_all = pd.concat(all_data, ignore_index=True)
            corr_all = df_all.corr(method=method)

            plt.figure(figsize=(8, 4))
            sns.heatmap(corr_all, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title("Full Population Parameter-Cost Correlation")
            plt.tight_layout()
            if save:
                plt.savefig(save+'.png', bbox_inches='tight')
        else:
            print("No valid individuals below the cost threshold for full population correlation.")

    def plot_partial_dependence(self, save=None):
        """
        Plots partial dependence curves for each optimization parameter using a surrogate model.

        Each plot shows how varying one parameter (while holding others constant) affects the predicted cost, 
        averaged over the population. This helps visualize parameter sensitivity and the surrogate model's learned 
        cost landscape.

        Parameters:
            save (str, optional): Path to save the resulting figure. If None, the plot is shown interactively.

        Returns:
            None
        """

        try:
            importlib.import_module('sklearn')
        except ImportError:
            print("sklearn package is required for the build_ancestry_graph function")
            return
        from sklearn.ensemble import RandomForestRegressor
        # Extract data
        parameters = []
        costs = []
        for ind in self.table.individuals:
            parameters.append(ind.parameters)
            costs.append(ind.cost)
        parameters = np.asarray(parameters)
        costs = np.asarray(costs)

        # Optional cost filtering
        mask = costs < self.parameters.badvalue
        parameters = parameters[mask]
        costs = costs[mask]

        n_params = parameters.shape[1]

        # Fit surrogate model
        model = RandomForestRegressor()
        model.fit(parameters, costs)

        # Generate partial dependence
        n_points = 50
        fig, axes = plt.subplots(n_params, 1, figsize=(8, 3 * n_params))

        if n_params == 1:
            axes = [axes]  # Make iterable

        for i in range(n_params):
            param_range = np.linspace(parameters[:, i].min(), parameters[:, i].max(), n_points)
            avg_preds = []

            for val in param_range:
                X_copy = parameters.copy()
                X_copy[:, i] = val  # Set i-th param to fixed value
                preds = model.predict(X_copy)
                avg_preds.append(np.mean(preds))

            axes[i].plot(param_range, avg_preds)
            axes[i].set_title(f"Partial Dependence: Param_{i}")
            axes[i].set_xlabel(f"Param_{i}")
            axes[i].set_ylabel("Avg Predicted Cost")
            axes[i].grid(True)

        plt.tight_layout()
        if save:
            fig.savefig(save, bbox_inches='tight')

    def plot_parameter_diversity(self, save=None):
        """
        Plots the diversity (standard deviation) of each parameter across generations.

        This visualizes how much variation exists in each optimization parameter over time, helping to assess 
        whether the population is converging or maintaining diversity. Low diversity may indicate premature convergence.

        Parameters:
            save (str, optional): Path to save the resulting figure. If None, the plot is shown interactively.

        Returns:
            None
        """
        # Compute parameter diversity over generations
        parameters = []
        for ind in self.table.individuals:
            parameters.append(ind.parameters)
        parameters = np.asarray(parameters)
        parameters = (parameters-np.min(parameters,axis=0))/(np.max(parameters,axis=0)-np.min(parameters,axis=0))
        param_diversity = []

        for pop in self.population:
            idx = pop.data['Individuals'].astype(int).values
            param_matrix = np.stack(parameters[idx,:])  # shape: (n_individuals, n_params)
            std_per_param = np.std(param_matrix, axis=0)
            param_diversity.append(std_per_param)

        # Convert to DataFrame
        df_diversity = pd.DataFrame(param_diversity, columns=[f"Param_{i}" for i in range(param_matrix.shape[1])])

        # Plot parameter diversity over generations
        plt.figure(figsize=(10, 6))
        for col in df_diversity.columns:
            plt.plot(df_diversity.index.to_numpy(), df_diversity[col].to_numpy(), label=col)

        plt.xlabel("Generation")
        plt.ylabel("Normalised Parameter Std Deviation")
        plt.title("Parameter Diversity Over Generations")
        plt.legend(title="Parameter")
        plt.grid(True)
        plt.tight_layout()
        if save:
            plt.savefig(save, bbox_inches='tight')
        
        try:
            importlib.import_module('seaborn')
            
        except ImportError:
            print("seaborn package is required for the build_ancestry_graph function")
            return
        import seaborn as sns

        # Reconstruct full distribution data for violin plots
        violin_data = []


        for gen_idx, pop in enumerate(self.population):
            idx = pop.data['Individuals'].astype(int).values
            param_matrix = parameters[idx, :]
            for param_idx in range(param_matrix.shape[1]):
                for val in param_matrix[:, param_idx]:
                    violin_data.append({
                        'Generation': gen_idx,
                        'Parameter': f'Param_{param_idx}',
                        'Value': val
                    })

        df_violin = pd.DataFrame(violin_data)
        # Create subplots
        n_params = param_matrix.shape[1]
        ncols = 4
        nrows = int(np.ceil(n_params/ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8), sharey=True)

        # Ensure axes is iterable
        if n_params == 1:
            axes = [axes]

        # Plot each parameter in its own subplot
        i=0
        for row in range(nrows):
            for col in range(ncols):
                if i<n_params:
                    sns.violinplot(
                        data=df_violin[df_violin["Parameter"] == f'Param_{i}'],
                        x="Generation",
                        y="Value",
                        inner="quart",
                        fill=False,
                        ax=axes[row,col]
                    )
                    axes[row,col].set_title(f"Parameter: Param_{i}")
                    axes[row,col].set_ylabel("Normalized Parameter Value")
                else:
                    axes[row,col].remove()
                i+=1

        plt.tight_layout()
        plt.suptitle("Parameter Distributions Across Generations", y=1.02)

        if save:
            fig.savefig(save.replace('.png', '_violin.png'), bbox_inches='tight')

    def plot_operations(self, save=None):
        """
        Plots a bar chart showing the relative frequency of genetic operations used.

        Parameters:
            save (str): If specified, path to save the figure.

        Returns:
            None
        """
        # Color map based on exploitation type
        base_colors = {
            'Random':'black','Elitism':'red','Replication':'lightcoral','Mutation':'indianred','Crossover':'darkred',
            'Reflection':'royalblue','Contraction':'lightsteelblue','Shrinked':'navy','Expanded':'skyblue',
            'Correction':'blueviolet','CorrectionRandom':'violet','CMA-ES':'royalblue','Scipy':'royalblue'
        }

        # Determine relevant colors based on ExploitationType
        if hasattr(self.parameters, 'ExploitationType'):
            exploitation = self.parameters.ExploitationType
            if exploitation == 'Downhill Simplex':
                relevant_ops = ['Random','Elitism','Replication','Mutation','Crossover',
                                'Reflection','Contraction','Shrinked','Expanded','Correction','CorrectionRandom']
            elif exploitation == 'CMA-ES':
                relevant_ops = ['Random','Elitism','Replication','Mutation','Crossover','CMA-ES']
            elif exploitation == 'Scipy':
                relevant_ops = ['Random','Elitism','Replication','Mutation','Crossover','Scipy']
            else:
                relevant_ops = list(base_colors.keys())
        else:
            relevant_ops = ['Random','Elitism','Replication','Mutation','Crossover']

        color_map = {op: base_colors[op] for op in relevant_ops}

        # Build a DataFrame of operation counts per generation
        op_counts_per_gen = []

        for pop in self.population:
            ops = pop.data[('Operation', 'Type')]
            count = pd.Series(ops).value_counts()
            count_all_ops = {op: count.get(op, 0) for op in relevant_ops}
            op_counts_per_gen.append(count_all_ops)

        df_counts = pd.DataFrame(op_counts_per_gen)
        df_normalized = df_counts.div(df_counts.sum(axis=1), axis=0).fillna(0)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        bottom = np.zeros(len(df_normalized))

        for op in relevant_ops:
            values = df_normalized[op].values
            ax.bar(range(len(df_normalized)), values, bottom=bottom, label=op, color=color_map[op])
            bottom += values

        ax.set_xlabel("Generation #")
        ax.set_ylabel("Relative Operation Frequency")
        ax.set_xticks(range(len(df_normalized)))
        ax.set_ylim(0, 1)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=5)

        if save:
            fig.savefig(save, bbox_inches='tight')

    def plot_J_evolution(self, fitness=False, save=None):
        """
        Plot cost evolution information for a genetic algorithm.

        Parameters:
            fitness (bool): If True, plots 1/cost instead of cost.
            save (str): If specified, path to save the figure.

        Returns:
            None
        """
        costs_per_gen = []
        bad_costs_per_gen = []
        medians = []
        minimums = []

        for pop in self.population:
            costs = np.array(pop.data['Costs'])
            if fitness:
                costs = 1 / costs
                bad_value_thresh = 1 / (self.parameters.badvalue / 1000)
                bad_costs = costs[costs > bad_value_thresh]
            else:
                bad_value_thresh = self.parameters.badvalue / 1000
                bad_costs = costs[costs < bad_value_thresh]

            costs_per_gen.append(costs)
            bad_costs_per_gen.append(bad_costs)
            medians.append(np.nanmedian(costs))
            minimums.append(np.nanmin(costs))

        medians = np.array(medians)
        minimums = np.array(minimums)

        min_bad = np.array([vals.min() if len(vals) else np.nan for vals in bad_costs_per_gen])
        max_bad = np.array([vals.max() if len(vals) else np.nan for vals in bad_costs_per_gen])

        # Plotting logic
        if hasattr(self.parameters, 'exploitation'):
            fig, axs = plt.subplots(1, 2, figsize=(10,6))
            # fig.suptitle("Convergence")

            # Left plot: median and minimum cost over generations
            axs[0].plot(range(1, len(minimums) + 1), minimums, 'r')
            axs[0].plot(range(1, len(medians) + 1), medians, 'k')
            axs[0].legend(['Best J', 'Median'])
            axs[0].set_xlabel('Generation #')
            axs[0].set_ylabel('1/J' if fitness else 'J')
            axs[0].set_xticks(range(1, len(medians) + 1))
            if np.all(minimums > 0) and np.min(minimums) > 1e-36:
                axs[0].set_yscale('log')

            min_ylim = float(np.nanmin(minimums))
            max_ylim = float(np.nanmax(medians))
            axs[0].set_ylim((min_ylim * 0.95, max_ylim * 1.05))

            # Right plot: individual costs with operation color
            ax = axs[1]
            if hasattr(self.parameters,'ExploitationType'):
                exploitation = self.parameters.ExploitationType
            else:
                exploitation = ''
                
            colors = {
                'Random': 'black', 'Elitism': 'red', 'Replication': 'lightcoral',
                'Mutation': 'indianred', 'Crossover': 'darkred'
            }
            if exploitation == 'Downhill Simplex':
                colors.update({
                    'Reflection': 'royalblue', 'Contraction': 'lightsteelblue',
                    'Shrinked': 'navy', 'Expanded': 'skyblue',
                    'Correction': 'blueviolet', 'CorrectionRandom': 'violet'
                })
            elif exploitation == 'CMA-ES':
                colors['CMA-ES'] = 'royalblue'
            elif exploitation == 'Scipy':
                colors['Scipy'] = 'royalblue'

            start = 0
            for pop in self.population:
                for i in range(pop.data.shape[0]):
                    cost = pop.data.loc[i, 'Costs']
                    op = pop.data.loc[i, ('Operation', 'Type')]
                    ax.scatter(i + start, cost, s=5, c=colors.get(op, 'gray'))
                start += pop.data.shape[0]
                ax.axvline(start, color='lightgrey', linestyle='--')

            if np.all(minimums > 0) and np.nanmin(minimums) > 1e-36:
                ax.set_yscale('log')
            ax.set_ylim((min_ylim * 0.95, np.nanmax(max_bad) * 1.05))
            ax.set_xlabel('Individuals')

            # Legend
            handles = [ax.scatter([], [], c=color, s=20, label=label) for label, color in colors.items()]
            ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(-0.1, 1.17), ncol=5, fancybox=True, shadow=True)

        else:
            fig, axs = plt.subplots(2)
            # fig.suptitle("Convergence")
            axs[0].plot(range(1, self.parameters.ngen + 1), np.min(costs_per_gen, axis=1), 'r')
            axs[0].plot(range(1, self.parameters.ngen + 1), np.median(costs_per_gen, axis=1), 'k')
            axs[0].legend(['Best J', 'Median'])
            axs[0].set_xlabel('Generation #')
            axs[0].set_ylabel('J')

            flat_costs = np.concatenate(costs_per_gen)
            flat_bad = np.concatenate([x for x in bad_costs_per_gen if len(x)])

            sc = axs[1].scatter(range(len(flat_costs)), flat_costs, s=5, c=flat_costs,
                                vmin=flat_bad.min() * 0.95, vmax=flat_bad.max() * 1.05)
            plt.colorbar(sc, ax=axs[1])
            axs[1].set_ylim((flat_bad.min() * 0.95, flat_bad.max() * 1.05))
            axs[1].set_xlabel('Individual #')
            axs[1].set_ylabel('J')

            start = 0
            for pop in self.population:
                start += pop.data.shape[0]
                axs[1].axvline(start, color='r', linestyle='--')

        if save:
            fig.savefig(save, bbox_inches='tight')