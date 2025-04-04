__author__ = 'Isaac Robledo Martín'
import numpy as np
import pandas as pd
import time
import os
import copy

from .individual import Individual
from .table import Table

from .tools.simplex import Simplex
from .tools.CMA_ES import CMA_ES
from .tools.api_scipy import API_Scipy

import re

# Function to filter strings based on invalid indices
def filter_valid_strings(strings, invalid_indices):
    # Regular expression to extract the integer at the end of the string
    pattern = re.compile(r"/Individual(\d+)$")
    valid_strings = []

    for s in strings:
        match = pattern.search(s)
        if match:
            # Extract the integer and convert it to an integer
            index = int(match.group(1))
            # Check if it's not in the list of invalid indices
            if index not in invalid_indices:
                valid_strings.append(s)

    return valid_strings

class Population(Simplex,CMA_ES,API_Scipy):
    """
    The Population class represents a population of individuals in a Genetic Algorithm.

    Attributes:
        - Nind (int): Number of individuals in the population.
        - generation (int): Current generation number.
        - repetition (int): Current repetition number.
        - state (str): State of the population ('None', 'Generated', 'Evaluated').
        - idx_to_evaluate (list): List to store indices of individuals to be evaluated.
        - idx_to_check (list): List to store indices of individuals to be checked for convergence.
        - data (pd.DataFrame): A DataFrame to store various information about the individuals in the population.

    Methods:
        - add_repetition(self): Adds a repetition to the data structure when individuals are evaluated multiple times.
        - display_info(self, rows=10): Displays information about the population.
        - generate_pop(self, HYGO_params, HYGO_table): Generates a new population based on specified parameters.
        - evolve_pop(self, HYGO_params, HYGO_table, new_size): Evolves the population to generate a new population.
        - crossover(self, old_pop, HYGO_params, HYGO_table): Performs crossover operation on the population.
        - elitism(self, old_pop, HYGO_params, HYGO_table): Performs elitism operation on the population.
        - mutation(self, old_pop, HYGO_params, HYGO_table): Performs mutation operation on the population.
        - replication(self, old_pop, HYGO_params, HYGO_table): Performs replication operation on the population.
        - sort_pop(self): Sorts the population based on costs in ascending order.
        - evaluate_population(self, idx_to_evaluate, HYGO_params, HYGO_table, path, simplex=False): Evaluates the population by calling the cost function for each individual.
        - compute_uncertainty(self, idx, rep): Computes uncertainty for a given individual.
        - check_params(self, params, HYGO_params): Checks if parameters are within the specified range and updates them if necessary.
    """

    def __init__(self, Nind, generation) -> None:

        '''
        Initialize the Population class.

        Parameters:
            - Nind (int): NUmber of individuals in the population
            - generation (int): generation number

        Attributes:
            - Nind (int): Number of individuals in the population.
            - generation (int): Current generation number.
            - repetition (int): Current repetition number.
            - state (str): State of the population ('None', 'Generated', 'Evaluated').
            - idx_to_evaluate (list): List to store indices of individuals to be evaluated.
            - idx_to_check (list): List to store indices of individuals to be checked for convergence.
        '''
        
        #Initialize the attributes
        self.Nind = Nind
        self.generation = generation
        self.repetition = 0
        self.state = 'None'
        self.idx_to_evaluate = []
        self.idx_to_check= []

        # Initialize data structure using pandas DataFrame with the appropriate size
        foo = (np.zeros(self.Nind)-1).tolist()
        foo_str = ['None']*self.Nind

        # Initialize the general columns of the data structure
        df = pd.MultiIndex.from_tuples([("Individuals",''),
                                        ("Costs",''),
                                        ("Uncertainty",'Minimum'), 
                                        ("Uncertainty",'All'), 
                                        ("Parents", "first"), 
                                        ("Parents", "second"),
                                        ('Operation','Type'),
                                        ('Operation','Point_parts')])
        
        # Create the data structure
        self.data = pd.DataFrame(columns=df)

        # Set placeholders in the general columns
        self.data['Individuals'] = foo
        self.data['Costs'] = foo
        self.data['Uncertainty','Minimum'] = foo
        self.data['Uncertainty','All'] = foo_str
        self.data['Parents','first'] = foo
        self.data['Parents','second'] = foo
        self.data['Operation','Type']  = foo_str
        self.data['Operation','Point_parts']  = foo_str

    def add_repetition(self):
        '''
        Add a repetition to the data structure.

        '''
        foo = (np.zeros(self.Nind)-1).tolist()
        foo_str = ['None']*self.Nind

        self.repetition += 1

        current_name = "Rep "+ str(self.repetition)

        self.data[current_name,'Evaluation_time']=foo
        self.data[current_name,'Path']=foo_str
        self.data[current_name,'Cost']=foo
        self.data[current_name,'Cost_terms']=foo_str    

    def display_info(self,rows=10):
        '''
        Display information about the population.

        Parameters:
            - rows (int): Number of rows to display in the DataFrame.

        '''
        print(self.data.info())
        print(self.data.head(rows))
        
    def generate_pop(self,HYGO_params,HYGO_table):
        '''
        Generate a new population.

        Parameters:
            - HYGO_params (object): An object containing parameters for the Genetic Algorithm.
            - HYGO_table (object): The table containing individuals.

        '''

        # Obtain the reference time
        TIME = time.time()

        if HYGO_params.verbose:
            print('################ Generating population ' + str(self.generation)+' ################')
            print('---Number of individuals in the population = '+str(self.Nind))

        initial = None

        # Generate initial population using Latin Hypercube Sampling if specified
        if HYGO_params.initialization == 'LatinHypercube':
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=HYGO_params.N_params)
            sample = sampler.random(n=HYGO_params.LatinN)
            bounds = np.array(HYGO_params.params_range)
            l_bounds = bounds[:,0]
            u_bounds = bounds[:,1]
            sample = qmc.scale(sample, l_bounds, u_bounds)
            initial = []
            # Check that the params are within the specified bounds
            for arr in sample:
                params = self.check_params(arr.tolist(),HYGO_params)
                initial.append(params)

        # Generate forced individuals if specified
        if HYGO_params.force_individuals:
            from .tools.individual_forced_generator import individual_forced_generator
            forced_params = individual_forced_generator(HYGO_params)
        else:
            forced_params = []

        # Combine initial and forced individuals
        if initial:
            if forced_params==[]:
                forced_params = initial
            else:
                forced_params=initial+forced_params

        indexes = []

        # Loop to generate individuals in the population
        for i in range(self.Nind):

            if HYGO_params.verbose:
                print('Generating individual ' + str(i+1) + '/' + str(self.Nind))

            checker = True
            counter = 1 
            # Handle duplicates if remove_duplicates is enabled by re-generating them
            #   until the maximum number of tries is reached
            while checker:
                new_ind = Individual()
                # Use specified parameters for the individual if available
                if i<len(forced_params) and counter==1:
                    new_ind.create(HYGO_params=HYGO_params,params=forced_params[i])
                else:
                    # Create random individual
                    new_ind.create(HYGO_params=HYGO_params)
                [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_ind)

                # Assume that the individual is valid
                valid = True

                # Check if the control is within bounds
                if HYGO_params.optimization == 'Control Law':
                    valid = [0]*HYGO_params.control_outputs
                    for j in range(HYGO_params.control_outputs):
                        valid[j] = int(np.sum(new_ind.ControlPoints[j,:]<np.array(HYGO_params.Control_range[j][0])) + np.sum(new_ind.ControlPoints[j,:]>np.array(HYGO_params.Control_range[j][1]))) == 0
                    valid = int(np.sum(np.array(valid))) == len(valid)

                # Check if there is a custom validity function
                if hasattr(HYGO_params,'validity') and callable(HYGO_params.validity):
                    custom_valid = HYGO_params.validity(new_ind.parameters)
                    valid = valid and custom_valid

                # Remove the individual if not valid
                if (not valid and not exists) and counter<HYGO_params.MaxTries and HYGO_params.remove_duplicates:
                    HYGO_table.remove_individual(int(idx))
                
                checker = HYGO_params.remove_duplicates and counter<HYGO_params.MaxTries and (exists or not valid)
                
                counter+=1

            if HYGO_params.optimization == 'Control Law' and HYGO_params.exploitation and HYGO_params.SimplexInterpolation:
                # Add the required attributes to the individual
                ind = HYGO_table.individuals[idx]
                ind.simplex_parents = [idx]
                ind.coefficients = [1]
                HYGO_table.individuals[idx] = ind

            indexes.append(idx)

            # Update the data DataFrame with individual information
            self.data.loc[i,'Individuals'] = idx
            self.data.loc[i,('Parents','first')]  = -1
            self.data.loc[i,('Parents','second')] = -1
            self.data.loc[i,('Operation','Type')] = 'Random'
            self.data.loc[i,('Operation','Point_parts')]  = 'None'
            
            # If it is not valid, assign a badvalue so it is not evaluated
            if not valid:
                self.data.loc[i,'Costs'] = HYGO_params.badvalue
                HYGO_table.individuals[idx].cost = HYGO_params.badvalue

        # Update the data DataFrame with the assigned indexes
        self.data['Individuals'] = indexes
        self.state = 'Generated'

        if HYGO_params.verbose:
            print('-->Generation created in ' + str(time.time()-TIME) + ' s')

    def evolve_pop(self,HYGO_params,HYGO_table,new_size):
        """
        Evolves the current population to generate a new population based on specified parameters.

        Parameters:
            - HYGO_params: An object containing parameters for the Genetic Algorithm.
            - HYGO_table: An instance of the Table class for storing individuals.
            - new_size: The desired size of the new population.

        Returns:
            - Population: A new Population object representing the evolved population.
        """
        from .tools.choose_operation import choose_operation

        # Create a new Population object for the evolved population
        new_pop = Population(new_size,self.generation+1)

        # Apply elitism to preserve the best individuals from the current population
        new_pop.elitism(old_pop=self,HYGO_params=HYGO_params,HYGO_table=HYGO_table)

        # Obtain the indexes of the individuals to be created through genetic operations
        indivs = new_pop.data['Individuals'].values.tolist()
        filled_indiv = len(indivs) - indivs.count(-1)

        if HYGO_params.verbose:
            print('################ Generating population ' + str(new_pop.generation)+' ################')
            print('---Number of individuals in the population = '+str(new_pop.Nind))
        
        # Loop until the new population is filled to the desired size
        while filled_indiv<(new_size):
            indivs = new_pop.data['Individuals'].values.tolist()
            filled_indiv = len(indivs) - indivs.count(-1)
            
            if HYGO_params.verbose:
                print('Generating individual ' + str(filled_indiv) + '/' + str(new_pop.Nind))
            
            # Choose the operation for generating a new individual (Replication, Mutation, or Crossover)
            operation = choose_operation(HYGO_params)

            if operation == 'Replication':
                # Perform the replication operation
                new_pop.replication(old_pop=self,HYGO_params=HYGO_params,HYGO_table=HYGO_table)
            elif operation == 'Mutation':
                # Perform the mutation operation
                new_pop.mutation(old_pop=self,HYGO_params=HYGO_params,HYGO_table=HYGO_table)
            else:
                # Perform the crossover operation if there is enough space
                if (len(indivs)-filled_indiv)>2:
                    new_pop.crossover(old_pop=self,HYGO_params=HYGO_params,HYGO_table=HYGO_table)

            # Update the number of individuals left to fill
            indivs = new_pop.data['Individuals'].values.tolist()
            filled_indiv = len(indivs) - indivs.count(-1)

        # Update the new population state
        new_pop.state = 'Generated'

        return new_pop

    def crossover(self,old_pop,HYGO_params,HYGO_table):
        """
        Applies crossover operation to generate new individuals in the population.

        Parameters:
            - old_pop: The previous generation's Population object from which individuals are taken to perform genetic operations.
            - HYGO_params: An object containing parameters for the Genetic Algorithm.
            - HYGO_table: An instance of the Table class for storing individuals.

        Returns:
            None
        """
        from .tools.select_individual import select_individual

        # Get the current index of individuals in the population
        indivs = self.data['Individuals'].values.tolist()
        current_idx = len(indivs) - indivs.count(-1)

        checker = True
        counter = 1

        # Check if there are enough individuals for crossover
        if (current_idx==len(indivs)) or (current_idx==len(indivs)-1):
            return

        # Loop until a valid crossover is achieved
        while checker:
            # Select two individuals from the previous generation
            idx1 = select_individual(HYGO_params,old_pop.Nind)
            idx2 = select_individual(HYGO_params,old_pop.Nind)

            # Obtain ther indexes in the old population
            idx_1 = old_pop.data.loc[idx1,'Individuals']
            idx_2 = old_pop.data.loc[idx2,'Individuals']

            # Obtain their objects from the Table
            ind1 = HYGO_table.individuals[int(idx_1)]
            ind2 = HYGO_table.individuals[int(idx_2)]

            # Perform crossover to generate two new individuals
            new_indiv1,new_indiv2,operation = Individual.crossover(HYGO_params,ind1,ind2)
            
            # Assume that the individual is valid
            valid1 = True
            valid2 = True

            # Check if the control is within bounds
            if HYGO_params.optimization == 'Control Law':
                valid1 = [0]*HYGO_params.control_outputs
                valid2 = [0]*HYGO_params.control_outputs
                for i in range(HYGO_params.control_outputs):
                    valid1[i] = int(np.sum(new_indiv1.ControlPoints[i,:]<np.array(HYGO_params.Control_range[i][0])) + np.sum(new_indiv1.ControlPoints[i,:]>np.array(HYGO_params.Control_range[i][1]))) == 0
                    valid2[i] = int(np.sum(new_indiv2.ControlPoints[i,:]<np.array(HYGO_params.Control_range[i][0])) + np.sum(new_indiv2.ControlPoints[i,:]>np.array(HYGO_params.Control_range[i][1]))) == 0
                valid1 = int(np.sum(np.array(valid1))) == len(valid1)
                valid2 = int(np.sum(np.array(valid2))) == len(valid2)
                # Check that the individuals are big enough
                length1 = ind1.chromosome.shape[0] >= HYGO_params.Minimum_instructions
                length2 = ind2.chromosome.shape[0] >= HYGO_params.Minimum_instructions

                valid1 = valid1 and length1
                valid2 = valid2 and length2
            
            # Check if there is a custom validity function
            if hasattr(HYGO_params,'validity') and callable(HYGO_params.validity):
                custom_valid1 = HYGO_params.validity(new_indiv1.parameters)
                valid1 = valid1 and custom_valid1
                custom_valid2 = HYGO_params.validity(new_indiv2.parameters)
                valid2 = valid2 and custom_valid2

            # Add the new individuals to the individual table
            [idx_n1,exists1] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_indiv1)
            [idx_n2,exists2] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_indiv2)

            # If removing duplicates, handle the cases where one of the new individuals exist
            if HYGO_params.remove_duplicates and counter<HYGO_params.MaxTries:

                # If 1 exists but 2 no, eliminate 2 in order to regenerate both individuals
                if exists1 and not exists2 and HYGO_params.remove_duplicates:
                    HYGO_table.remove_individual(int(idx_n2))

                # Same as before
                if exists2 and not exists1 and HYGO_params.remove_duplicates:
                    HYGO_table.remove_individual(int(idx_n1))

                # Check if the individuals are within control range
                if not exists1 and not exists2 and (not valid1 or not valid2) and HYGO_params.remove_duplicates:
                    HYGO_table.remove_individual(int(max([idx_n1,idx_n2])))
                    HYGO_table.remove_individual(int(min([idx_n1,idx_n2])))

                # If both exist they will not be added to the table and checker will be True
                
                checker = HYGO_params.remove_duplicates and counter<HYGO_params.MaxTries and (exists1 or exists2 or not valid1 or not valid2)

            else:
                checker = False
            counter +=1

        if HYGO_params.optimization == 'Control Law' and HYGO_params.exploitation and HYGO_params.SimplexInterpolation:
            # Add the required attributes to the individual
            ind1 = HYGO_table.individuals[idx_n1]
            ind1.simplex_parents = [idx_n1]
            ind1.coefficients = [1]
            HYGO_table.individuals[idx_n1] = ind1
            ind2 = HYGO_table.individuals[idx_n2]
            ind2.simplex_parents = [idx_n2]
            ind2.coefficients = [1]
            HYGO_table.individuals[idx_n2] = ind2
        
        # Update the population's data with the information of the new individuals
        if exists1:
            self.data.loc[[current_idx],['Costs']] = HYGO_table.individuals[int(idx_n1)].cost

        if exists2:
            self.data.loc[[current_idx+1],['Costs']] = HYGO_table.individuals[int(idx_n2)].cost

        # If they are not valid, assign a badvalue so they are not evaluated
        if not valid1 and not exists1:
            self.data.loc[[current_idx],['Costs']] = HYGO_params.badvalue
            HYGO_table.individuals[idx_n1].cost = HYGO_params.badvalue
        if not valid2 and not exists2:
            self.data.loc[[current_idx+1],['Costs']] = HYGO_params.badvalue
            HYGO_table.individuals[idx_n2].cost = HYGO_params.badvalue
        
        self.data.loc[current_idx,'Individuals'] = int(idx_n1)
        self.data.loc[current_idx,('Parents','first')] = int(idx_1)
        self.data.loc[current_idx,('Parents','second')] = int(idx_2)
        self.data.loc[current_idx,('Operation','Type')] = 'Crossover'
        self.data.loc[current_idx,('Operation','Point_parts')] = str(operation[0])

        self.data.loc[current_idx+1,'Individuals'] = int(idx_n2)
        self.data.loc[current_idx+1,('Parents','first')] = int(idx_1)
        self.data.loc[current_idx+1,('Parents','second')] = int(idx_2)
        self.data.loc[current_idx+1,('Operation','Type')] = 'Crossover'
        self.data.loc[current_idx+1,('Operation','Point_parts')] = str(operation[0])

    def elitism(self,old_pop,HYGO_params,HYGO_table):
        """
        Applies elitism to select the top individuals from the previous generation.

        Parameters:
            - old_pop: The previous generation's Population object.
            - HYGO_params: An instance of the HYGO_params class containing parameters for the Genetic Algorithm.
            - HYGO_table: An instance of the Table class for storing individuals.

        Returns:
            None
        """
        # Number of individuals to be preserved through elitism
        n_elitism = HYGO_params.N_elitism

        # Copy the top individuals' information from the previous generation to the new generation
        self.data.loc[0:n_elitism-1,['Individuals','Costs','Uncertainty']] = old_pop.data.loc[0:n_elitism-1,
                                    ['Individuals','Costs','Uncertainty']]
        
        # Get the indices of the top individuals
        idx = self.data.loc[0:n_elitism-1,['Individuals']]
        idx = idx.values.tolist()

        # Update ocurrences and operation information for the top individuals
        counter = 0
        for i in idx:
            HYGO_table.individuals[int(i[0])].ocurrences +=1
            self.data.loc[counter,('Parents','first')] = int(i[0])
            self.data.loc[counter,('Parents','second')] = -1
            self.data.loc[counter,('Operation','Type')] = 'Elitism'
            self.data.loc[counter,('Operation','Point_parts')] = str('None')

            counter +=1

    def mutation(self,old_pop,HYGO_params,HYGO_table):
        """
        Perform mutation on individuals in the population.

        Parameters:
            - old_pop (Population): The previous generation of the population.
            - HYGO_params (object): An object containing parameters for the Genetic Algorithm.
            - HYGO_table (Table): A table object storing individuals and their information.

        Returns:
            None
        """
        from .tools.select_individual import select_individual

        # Get the index to be filled
        indivs = self.data['Individuals'].values.tolist()
        current_idx = len(indivs) - indivs.count(-1)

        checker = True # Flag to check if a valid mutation is found
        counter = 1 # Counter for limiting the number of attempts to find a valid mutation

        while checker:
            # Select an individual for mutation
            idx1 = select_individual(HYGO_params,old_pop.Nind)
            idx_1 = old_pop.data.loc[idx1,'Individuals']

            # Create a new individual with mutated chromosome
            ind1 = HYGO_table.individuals[int(idx_1)]

            # Add the mutated individual to the population
            mind,instructions = Individual.mutate(HYGO_params,ind1)

            # Check for duplicates if enabled in the algorithm parameters
            [midx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=mind)

            # Assume that the individual is valid
            valid = True

            # Check if the control is within bounds
            if HYGO_params.optimization == 'Control Law':
                valid = [0]*HYGO_params.control_outputs
                for i in range(HYGO_params.control_outputs):
                    valid[i] = int(np.sum(mind.ControlPoints[i,:]<np.array(HYGO_params.Control_range[i][0])) + np.sum(mind.ControlPoints[i,:]>np.array(HYGO_params.Control_range[i][1]))) == 0
                valid = int(np.sum(np.array(valid))) == len(valid)

            # Check if there is a custom validity function
            if hasattr(HYGO_params,'validity') and callable(HYGO_params.validity):
                custom_valid = HYGO_params.validity(mind.parameters)
                valid = valid and custom_valid

            # Remove the individual if not valid
            if (not valid and not exists) and counter<HYGO_params.MaxTries and HYGO_params.remove_duplicates:
                HYGO_table.remove_individual(int(midx))
            
            checker = HYGO_params.remove_duplicates and counter<HYGO_params.MaxTries and (exists or not valid)

            counter+=1

        if HYGO_params.optimization == 'Control Law' and HYGO_params.exploitation and HYGO_params.SimplexInterpolation:
            # Add the required attributes to the individual
            ind = HYGO_table.individuals[midx]
            ind.simplex_parents = [midx]
            ind.coefficients = [1]
            HYGO_table.individuals[midx] = ind

        # Update the population data with information about the mutated individual
        if exists:
            self.data.loc[current_idx,['Costs']] = HYGO_table.individuals[midx].cost

        # If it is not valid, assign a badvalue so it is not evaluated
        if not valid and not exists:
            self.data.loc[[current_idx],['Costs']] = HYGO_params.badvalue
            HYGO_table.individuals[midx].cost = HYGO_params.badvalue

        self.data.loc[current_idx,'Individuals'] = midx
        self.data.loc[current_idx,('Parents','first')] = int(idx_1)
        self.data.loc[current_idx,('Parents','second')] = -1
        self.data.loc[current_idx,('Operation','Type')] = 'Mutation'
        self.data.loc[current_idx,('Operation','Point_parts')] = str(instructions)

    def replication(self,old_pop,HYGO_params,HYGO_table):
        """
        Perform replication on individuals in the population.

        Parameters:
            - old_pop (Population): The previous generation of the population.
            - HYGO_params (object): An object containing parameters for the Genetic Algorithm.
            - HYGO_table (Table): A table object storing individuals and their information.

        Returns:
            None
        """
        from .tools.select_individual import select_individual

        # Get the index to be filled
        indivs = self.data['Individuals'].values.tolist()
        current_idx = len(indivs) - indivs.count(-1)

        # Select an individual for replication
        idx1 = select_individual(HYGO_params,old_pop.Nind)
        idx_1 = old_pop.data.loc[idx1,'Individuals']

        # Copy information from the selected individual to the current population
        self.data.loc[current_idx,['Individuals','Costs','Uncertainty']] = old_pop.data.loc[int(idx1),
                                    ['Individuals','Costs','Uncertainty']]
        
        HYGO_table.individuals[int(idx_1)].ocurrences +=1
        self.data.loc[current_idx,('Parents','first')] = int(idx_1)
        self.data.loc[current_idx,('Parents','second')] = -1
        self.data.loc[current_idx,('Operation','Type')] = 'Replication'
        self.data.loc[current_idx,('Operation','Point_parts')] = str('None')

    def sort_pop(self):
        '''
        Sort the population according to the costs in ascending order
        '''
        self.data = self.data.sort_values(by=['Costs'],ignore_index=True)

    def evaluate_population(self,idx_to_evaluate,HYGO_params,HYGO_table,path,exploitation=False):
        """
        Evaluate the individuals in the population, considering the specified number of repetitions.

        Parameters:
            - idx_to_evaluate (list): List of indices representing individuals to be evaluated. A list shall be
                provided for each repetition.
            - HYGO_params (object): An object containing parameters for the Genetic Algorithm.
            - HYGO_table (Table): A table object storing individuals and their information.
            - path (str): The path where the evaluation results will be stored.
            - exploitation (bool): Flag indicating whether the evaluation is part of the exploitation method.

        Returns:
            - HYGO_table (Table): The updated table of individuals after evaluation.
            - checker (bool): A flag indicating whether the convergence by the number of individuals is reached.
        """
        checker = True #It only changes if the convergence by number of individuals is turned on
                       # and that upper limit is reached 
        # Display information about the evaluation if verbose mode is enabled
        if HYGO_params.verbose and not exploitation:
            print('################ Evaluation of generation '+str(self.generation)+' ################')
        path = path + '/Gen'+str(self.generation)
        
        # Create the directory for the generation if specified
        if HYGO_params.individual_paths or HYGO_params.security_backup:
            if not os.path.isdir(path):
                os.mkdir(path)

        # Evaluate the specified number of repetitions
        for rep in range(HYGO_params.repetitions):
            # Obtain the individuals indexes
            indivs_idx = self.data.loc[idx_to_evaluate[rep],'Individuals']
            indivs_idx = indivs_idx.values.tolist()
            indivs=[]
            
            # Obtain the individuals objects
            for idx in indivs_idx:
                indivs.append(HYGO_table.individuals[int(idx)])
            
            if HYGO_params.verbose and not exploitation:
                print('\n--> Starting repetition '+str(rep+1))
            
            current_rep_name = "Rep "+ str(rep+1) #Repetition name
            current_path = path+'/Rep'+str(rep+1) #Repetition folder
            
            # Add a repetition if the individuals to be evaluated are not from the exploitation
            if not exploitation and ("Rep "+ str(rep+1)) not in self.data.columns:
                self.add_repetition() #Add repetition
                
            # Create the repetition path
            if HYGO_params.individual_paths:
                if not os.path.isdir(current_path):
                    os.mkdir(current_path) #Create directory if does not exist and params option selected

            # Obtain the number of individuals to evaluate
            ninds = len(indivs)
            # Obtain batch size
            if hasattr(HYGO_params,'batch_size') and hasattr(HYGO_params,'batch_evaluation') and HYGO_params.batch_evaluation:
                batch_size = HYGO_params.batch_size
            else:
                # If not specified individuals will be evaluated 1 by one
                batch_size = 1

            # Loop for obtaining the cost for each individual
            for i in range(0,ninds,batch_size):
                # Check that the current batch will not be longer than the number of in individuals
                if (i+batch_size)>ninds:
                    batch_size = ninds-i

                # Check if the max number evaluation is reached
                if HYGO_params.check_type=='Neval' or HYGO_params.limit_evaluations:
                    for j in idx_to_evaluate[rep][i:i+batch_size]:
                        if int(self.data.loc[j,'Individuals'])>=HYGO_params.neval:
                            checker = False
                            continue

                if HYGO_params.verbose:
                    print('Evaluating individuals '+str(idx_to_evaluate[rep][i:i+batch_size]).replace('[','').replace(']',''))
                    
                # Create the individuals path
                ind_paths=[]
                for j in idx_to_evaluate[rep][i:i+batch_size]:
                    ind_path = current_path+'/Individual'+str(j)
                    ind_paths.append(ind_path)
                    if HYGO_params.individual_paths:
                        if not os.path.isdir(ind_path):
                            os.mkdir(ind_path)

                # Obtain an evaluation time reference
                ref = time.time()

                # Create a sub-list of the indexes evaluated in this batch
                not_valid_idx = []
                not_valid_inds = [] # TODO CHANGED

                # Store individual's params
                params = []
                for j in range(i,i+batch_size):
                    # Obtain the individual to get the parameters
                    ind = indivs[j]

                    # Check if it has to be evaluated
                    if float(pd.isna(self.data.loc[idx_to_evaluate[rep][j],'Costs']))==0 and not abs(float(self.data.loc[idx_to_evaluate[rep][j],'Costs'])-(-1))<1e-9:
                        if HYGO_params.verbose:
                            print('Individual '+str(idx_to_evaluate[rep][j])+ ' not evaluated, already has an assigned cost')
                        self.data.loc[idx_to_evaluate[rep][j],(current_rep_name,'Cost_terms')] = str(np.nan)
                        self.data.loc[idx_to_evaluate[rep][j],(current_rep_name,'Evaluation_time')] = np.nan
                        self.data.loc[idx_to_evaluate[rep][j],('Uncertainty','Minimum')] = np.nan
                        self.data.loc[idx_to_evaluate[rep][j],('Uncertainty','All')] = np.nan
                        HYGO_table.individuals[int(indivs_idx[j])].cost = HYGO_params.badvalue
                        # Update data in the population
                        if HYGO_params.individual_paths:
                            self.data.loc[idx_to_evaluate[rep][j],(current_rep_name,'Path')] = ind_paths[j-i] #TODO: changed
                            HYGO_table.individuals[int(indivs_idx[j])].path.append(ind_paths[j-i])

                        # Identify individuals that are not valid
                        not_valid_idx.append(j)
                        not_valid_inds.append(idx_to_evaluate[rep][j]) # TODO CHANGED

                        # Make a security backup if specified
                        if HYGO_params.security_backup:
                            import dill
                            file = open(path+'/pop_backup.obj','wb')
                            dill.dump(self,file)
                            file.close()
                            file = open(path+'/table_backup.obj','wb')
                            dill.dump(HYGO_table,file)
                            file.close()
                    else:
                        # Obtain the individual parameters
                        params.append(ind.parameters)
                
                ind_paths = filter_valid_strings(ind_paths, not_valid_inds) # TODO CHANGED

                if params!=[]:
                    if len(params)==1:
                        # Obtain the cost for the individual
                        if HYGO_params.individual_paths: 
                            output = HYGO_params.cost_function(params[0],ind_paths[0])
                        else:
                            output = HYGO_params.cost_function(params[0])
                    else:
                        # Obtain the cost for the individual
                        if HYGO_params.individual_paths: 
                            output = HYGO_params.cost_function(params,ind_paths)
                        else:
                            output = HYGO_params.cost_function(params)
                else:
                    output = []
                    
                # Obtain evaluation time
                eval_time = time.time()-ref 
                
                if params!=[]:
                    if len(output)>2:
                        raise ValueError('No more than 2 outputs are allowed for a cost function definition')
                    else:
                        J = output[0]
                        J_vals = output[1]
                        if len(params)==1:
                            J = [J]
                            J_vals = [J_vals]

                pos = 0
                for j in range(i,i+batch_size):
                    if j in not_valid_idx:
                        # Eliminate the index from the corresponding list
                        self.idx_to_evaluate[rep].pop(0)
                    else:
                        self.data.loc[idx_to_evaluate[rep][j],(current_rep_name,'Cost')] = J[pos]
                        self.data.loc[idx_to_evaluate[rep][j],(current_rep_name,'Cost_terms')] = str(J_vals[pos])
                        self.data.loc[idx_to_evaluate[rep][j],(current_rep_name,'Evaluation_time')] = eval_time

                        # Add the index for latter uncertainty check
                        self.idx_to_check.append(idx_to_evaluate[rep][j])
                        self.idx_to_check = np.unique(self.idx_to_check).tolist()

                        # Update data in the population
                        if HYGO_params.individual_paths:
                            self.data.loc[idx_to_evaluate[rep][j],(current_rep_name,'Path')] = ind_paths[pos]
                            HYGO_table.individuals[int(indivs_idx[j])].path.append(ind_paths[pos])

                        # Update position of the cost value
                        pos += 1
                        
                        # Eliminate the index from the corresponding list
                        self.idx_to_evaluate[rep].pop(0)
                        
                        # Make a security backup if specified
                        if HYGO_params.security_backup:
                            import dill
                            file = open(path+'/pop_backup.obj','wb')
                            dill.dump(self,file)
                            file.close()
                            file = open(path+'/table_backup.obj','wb')
                            dill.dump(HYGO_table,file)
                            file.close()

        # Obtain the indexes to be checked for uncertainty
        idx_check = np.unique(self.idx_to_check)
        idx_to_evaluate_new=[]
        
        # Compute uncertainties
        for idx in idx_check:
            minun,valid_cost,uncertainties = self.compute_uncertainty(idx,HYGO_params.repetitions)
            
            # Save the data in the population
            self.data.loc[idx,('Uncertainty','Minimum')] = minun
            self.data.loc[idx,('Uncertainty','All')] = str(uncertainties)
            self.data.loc[idx,'Costs'] = valid_cost #Update pop costs
            
            # Check individuals outside uncertainty
            if HYGO_params.repetitions>1 and minun>HYGO_params.uncertainty:
                # Add to the list o indexes to evaluate of the extra repetition
                idx_to_evaluate_new.append(idx)
            
            # Eliminate the individual from the check for uncertainty list
            self.idx_to_check.pop(0)

        additional_rep = False
        non_valid_idx = []
        
        # Last repetition for individuals outside uncertainty
        if HYGO_params.repetitions>1 and HYGO_params.repeat_indivs_outside_uncertainty:
            if len(self.idx_to_evaluate)==(HYGO_params.repetitions+1):
                # Obtain the list of indexes to evaluate corresponding to the extra repetition
                idx_to_evaluate_old = idx_to_evaluate[-1]
                # Eliminate redundant indexes
                idx_to_eval = np.unique(idx_to_evaluate_new + idx_to_evaluate_old)
                
                #Save the idx to evaluate
                self.idx_to_evaluate[HYGO_params.repetitions] += copy.deepcopy(idx_to_eval.tolist())
            else:
                idx_to_eval = np.array(idx_to_evaluate_new)
                self.idx_to_evaluate.append(copy.deepcopy(idx_to_eval.tolist()))
            
            # Loop to evaluate the individuals outside the uncertainty
            if len(idx_to_eval)>0:
                # Set the flag for the additional repetition to True
                additional_rep = True
                
                # Add repetition if the evaluation corresponds to GA
                if not exploitation:
                    self.add_repetition()
                else:
                    nreps = HYGO_params.repetitions
                    cols = self.data.columns
                    # If the evaluation is for the exploitation individuals, check that the repetition
                    #   exists and if not, add it
                    if ('Rep '+str(nreps+1),'Cost') not in cols:
                        self.add_repetition()

                if HYGO_params.verbose and not exploitation:
                    print('\n--> Starting repetition '+str(self.repetition))
                
                # Obtain the individuals indexes
                indivs_idx = self.data.loc[idx_to_eval,'Individuals']
                indivs_idx=indivs_idx.values.tolist()
                indivs=[]
                
                # Obtain the individuals' objects
                for idx in indivs_idx:
                    indivs.append(HYGO_table.individuals[int(idx)])

                # Create the repetition directory if specified
                rep = self.repetition
                current_rep_name = "Rep "+ str(rep)
                current_path = path+'/Rep'+str(rep)
                if HYGO_params.individual_paths:
                    if not os.path.isdir(current_path):
                        os.mkdir(current_path)
                
                #################################
                # Obtain the number of individuals to evaluate
                ninds = len(indivs)
                for i in range(0,ninds,batch_size):
                    # Check that the current batch will not be longer than the number of in individuals
                    if (i+batch_size)>ninds:
                        batch_size = ninds-i

                    if HYGO_params.verbose:
                        print('Evaluating individuals '+str(idx_to_eval[i:i+batch_size]).replace('[','').replace(']',''))
                        
                    # Create the individuals path
                    ind_paths=[]
                    for j in idx_to_eval[i:i+batch_size]:
                        ind_path = current_path+'/Individual'+str(j)
                        ind_paths.append(ind_path)
                        if HYGO_params.individual_paths:
                            if not os.path.isdir(ind_path):
                                os.mkdir(ind_path)

                    # Obtain an evaluation time reference
                    ref = time.time()

                    # Create a sub-list of the indexes evaluated in this batch
                    not_valid_idx = []
                    not_valid_inds = [] # TODO CHANGED

                    # Store individual's params
                    params = []
                    for j in range(i,i+batch_size):
                        # Obtain the individual to get the parameters
                        ind = indivs[j]

                        # Check if it has to be evaluated
                        if float(pd.isna(self.data.loc[idx_to_eval[j],'Costs']))==0 and not abs(float(self.data.loc[idx_to_eval[j],'Costs'])-(-1))<1e-9:
                            if HYGO_params.verbose:
                                print('Individual '+str(idx_to_eval[j])+ ' not evaluated, already has an assigned cost')
                            self.data.loc[idx_to_eval[j],(current_rep_name,'Cost_terms')] = str(np.nan)
                            self.data.loc[idx_to_eval[j],(current_rep_name,'Evaluation_time')] = np.nan
                            self.data.loc[idx_to_eval[j],('Uncertainty','Minimum')] = np.nan
                            self.data.loc[idx_to_eval[j],('Uncertainty','All')] = np.nan
                            HYGO_table.individuals[int(idx_to_eval[j])].cost = HYGO_params.badvalue
                            # Update data in the population
                            if HYGO_params.individual_paths:
                                self.data.loc[idx_to_eval[j],(current_rep_name,'Path')] = ind_paths[j]
                                HYGO_table.individuals[int(indivs_idx[j])].path.append(ind_paths[j])

                            # Identify individuals that are not valid
                            not_valid_idx.append(j)
                            not_valid_inds.append(idx_to_evaluate[rep][j]) # TODO CHANGED

                            # Make a security backup if specified
                            if HYGO_params.security_backup:
                                import dill
                                file = open(path+'/pop_backup.obj','wb')
                                dill.dump(self,file)
                                file.close()
                                file = open(path+'/table_backup.obj','wb')
                                dill.dump(HYGO_table,file)
                                file.close()
                        else:
                            # Obtain the individual parameters
                            params.append(ind.parameters)

                    ind_paths = filter_valid_strings(ind_paths, not_valid_inds) # TODO CHANGED

                    if len(params)==1:
                        # Obtain the cost for the individual
                        if HYGO_params.individual_paths: 
                            output = HYGO_params.cost_function(params[0],ind_paths[0])
                        else:
                            output = HYGO_params.cost_function(params[0])
                    else:
                        # Obtain the cost for the individual
                        if HYGO_params.individual_paths: 
                            output = HYGO_params.cost_function(params,ind_paths)
                        else:
                            output = HYGO_params.cost_function(params)
                        
                    # Obtain evaluation time
                    eval_time = time.time()-ref 
                    
                    if len(output)>2:
                        raise ValueError('No more than 2 outputs are allowed for a cost function definition')
                    else:
                        J = output[0]
                        J_vals = output[1]
                        if len(params)==1:
                            J = [J]
                            J_vals = [J_vals]
                    pos = 0
                    for j in range(i,i+batch_size):
                        if j in not_valid_idx:
                            # Eliminate the index from the corresponding list
                            self.idx_to_evaluate[rep].pop(0)
                        else:
                            self.data.loc[idx_to_eval[j],(current_rep_name,'Cost')] = J[pos]
                            self.data.loc[idx_to_eval[j],(current_rep_name,'Cost_terms')] = str(J_vals[pos])
                            self.data.loc[idx_to_eval[j],(current_rep_name,'Evaluation_time')] = eval_time
                            # Update position of the cost value
                            pos += 1

                            # Add the index for latter uncertainty check
                            self.idx_to_check.append(idx_to_eval[j])
                            self.idx_to_check = np.unique(self.idx_to_check).tolist()

                            # Update data in the population
                            if HYGO_params.individual_paths:
                                self.data.loc[idx_to_eval[j],(current_rep_name,'Path')] = ind_paths[j]
                                HYGO_table.individuals[int(indivs_idx[j])].path.append(ind_paths[j])

                            # Eliminate the index from the corresponding list
                            self.idx_to_evaluate[HYGO_params.repetitions].pop(0)
                            
                            # Make a security backup if specified
                            if HYGO_params.security_backup:
                                import dill
                                file = open(path+'/pop_backup.obj','wb')
                                dill.dump(self,file)
                                file.close()
                                file = open(path+'/table_backup.obj','wb')
                                dill.dump(HYGO_table,file)
                                file.close()

            # Obtain the indexes to be checked for uncertainty
            idx_check = np.unique(self.idx_to_check)
            non_valid_idx = []
            
            # Check uncertainty of the extra repetition
            for idx in idx_check:
                # Compute uncertainties including the extra repetition if it took place
                if additional_rep:
                    minun,valid_cost,uncertainties = self.compute_uncertainty(idx,HYGO_params.repetitions+1)
                else:
                    minun,valid_cost,uncertainties = self.compute_uncertainty(idx,HYGO_params.repetitions)
                
                # Save the data in the population
                self.data.loc[idx,('Uncertainty','Minimum')] = minun
                self.data.loc[idx,('Uncertainty','All')] = str(uncertainties)
                self.data.loc[idx,'Costs'] = valid_cost #Update pop costs
                
                # Check individuals outside uncertainty
                if minun>HYGO_params.uncertainty:
                    non_valid_idx.append(idx)
                    
                # Eliminate the individual from the check for uncertainty list
                self.idx_to_check.pop(0)

        non_valid_idx = np.array(non_valid_idx)

        # Assign bad value costs to the individuals outside of the uncertainty
        if non_valid_idx.size>0:
            self.data.loc[non_valid_idx,'Costs'] = HYGO_params.badvalue
            for idx in non_valid_idx:
                HYGO_table.individuals[int(idx)].cost = HYGO_params.badvalue

        idx_to_drop=[]
        # Eliminate the rows corresponding to the individuals that were not evaluated due to the 
        #   maximum numbr of evaluations
        if (HYGO_params.check_type=='Neval' or HYGO_params.limit_evaluations) and not checker:
            for idx in range(self.data.shape[0]):
                if int(self.data.loc[idx,'Individuals'])>=HYGO_params.neval:
                    HYGO_table.remove_individual(-1)
                    idx_to_drop.append(int(idx))
            self.data = self.data.drop(idx_to_drop)
            self.data=self.data.reset_index(drop=True)

        # Update the population's state
        self.state = 'Evaluated'

        # Obtain the evaluated individuals indexes
        idx = self.data['Individuals']
        
        # Update the table data
        for i,j in enumerate(idx):
            HYGO_table.individuals[int(j)].cost = float(self.data.loc[i,'Costs'])
            HYGO_table.costs[int(j)] = float(self.data.loc[i,'Costs'])

        # Sort the population
        self.sort_pop()
        
        # Make a security backup if specified
        if HYGO_params.security_backup:
            import dill
            file = open(path+'/pop_backup.obj','wb')
            dill.dump(self,file)
            file.close()
            file = open(path+'/table_backup.obj','wb')
            dill.dump(HYGO_table,file)
            file.close()

        return HYGO_table,checker

    def compute_uncertainty(self,idx,rep):
        """
        Compute the uncertainty for a given individual based on the minimum cost among repetitions.

        Parameters:
            - idx (int): Index of the individual in the population.
            - reps (int): Number of repetitions for evaluation.

        Returns:
            - minun (float): Minimum uncertainty among repetitions.
            - valid_cost (float): cost corresponding to the average between the repetitions
                with least uncertainty between them.
            - uncertainties (dict): Dictionary of uncertainties.
        """
        
        minun = 1e36

        # Initialize dictionary to store uncertainties for each repetition
        uncertainty = {}

        # Iterate over repetitions
        if rep>1:
            for i in range(rep):
                for j in range(rep):
                    if j>i:
                        rep_name1 = 'Rep ' + str(i+1)
                        rep_name2 = 'Rep ' + str(j+1)
                        
                        # Obtain the cost values for the repetitions
                        vals = [self.data.loc[idx,(rep_name1,'Cost')],self.data.loc[idx,(rep_name2,'Cost')]]
                        
                        # Compute the uncertainty
                        un = (max(vals)-min(vals))/max(vals)
                        
                        # Store the value in the dictionary
                        uncertainty[str(i)+'-'+str(j)]=np.abs(un)
                        
                        # Update the minimu uncertainty value and cost
                        if un<minun or np.isnan(un):
                            minun=un
                            valid_cost = np.mean(vals)
        else:
            minun=0
            valid_cost = self.data.loc[idx,('Rep 1','Cost')]

        return minun,valid_cost,uncertainty

    def check_params(self,params,HYGO_params):
        """
        Check and adjust the parameters to ensure they are within the specified bounds and granularity.

        Parameters:
            - params (list): List of parameters to be checked and adjusted.
            - HYGO_params: An object containing parameters for the Genetic Algorithm.

        Returns:
            - adjusted_params (list): List of adjusted parameters.
        """
        # Iterate over parameters
        for i,param in enumerate(params):
            # Check if the parameter is below the lower bound
            if param<HYGO_params.params_range[i][0]:
                params[i]=HYGO_params.params_range[i][0]
            
            # Check if the parameter is above the upper bound
            elif param>HYGO_params.params_range[i][1]:
                params[i]=HYGO_params.params_range[i][1]
            
            else:
                Nb_bits = HYGO_params.Nb_bits
                
                # Determine the step size based on the number of bits
                if type(Nb_bits)!=int:
                    if len(Nb_bits)>1:
                        dx = (HYGO_params.params_range[i][1]-HYGO_params.params_range[i][0])/(2**Nb_bits[i]-1)
                    else:
                        dx = (HYGO_params.params_range[i][1]-HYGO_params.params_range[i][0])/(2**Nb_bits[0]-1)
                else:
                    dx = (HYGO_params.params_range[i][1]-HYGO_params.params_range[i][0])/(2**Nb_bits-1)

                # Check the granularity and adjust the parameter if needed
                checker = round(np.mod((param-HYGO_params.params_range[i][0]),dx)/dx)

                if checker==0:
                    param = float(param - np.mod((param-HYGO_params.params_range[i][0]),dx))
                elif checker==1:
                    param = float(param + (dx - np.mod((param-HYGO_params.params_range[i][0]),dx)))
                
                params[i] = param

        return params
