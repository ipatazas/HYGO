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
    
    def evaluate_and_store(self, HYGO_params, HYGO_table, base_path, rep, indices, exploitation):
        """
        Evaluate a batch of individuals in a given repetition, store results, and handle directories, backups, and limits.

        Parameters:
            HYGO_params: GA configuration object.
            HYGO_table: Table containing individuals.
            base_path: Directory to store evaluations.
            rep: Repetition index.
            indices: List of individual row indices to evaluate.
            exploitation: Whether this is exploitation evaluation.

        Returns:
            Tuple[HYGO_table, bool]: Updated table and evaluation continuation flag.
        """
        rep_name = f"Rep {rep+1}"
        rep_path = os.path.join(base_path, f"Rep{rep+1}")
        if HYGO_params.individual_paths:
            os.makedirs(rep_path, exist_ok=True)

        if not exploitation and ((rep_name, 'Cost') not in self.data.columns):
            self.add_repetition()

        indivs_idx = self.data.loc[indices, 'Individuals'].values.tolist()
        indivs = [HYGO_table.individuals[int(i)] for i in indivs_idx]
        ninds = len(indivs)
        batch_size = HYGO_params.batch_size if getattr(HYGO_params, 'batch_evaluation', False) else 1

        for i in range(0, ninds, batch_size):
            actual_size = min(batch_size, ninds - i)
            batch_inds = indivs[i:i+actual_size]
            batch_idxs = indices[i:i+actual_size]
            valid_idxs, valid_params, valid_paths, invalid_map = [], [], [], {}
            ref_time = time.time()

            # Check and collect valid individuals in batch
            for k, j in enumerate(batch_idxs):
                # Evaluation limit enforcement
                if HYGO_params.check_type == 'Neval' or HYGO_params.limit_evaluations:
                    idx = int(self.data.loc[j,'Individuals'])
                    if idx >= HYGO_params.neval:
                        print(f"[Max Eval] Individual {idx} exceeded evaluation limit ({HYGO_params.neval}).")
                        return HYGO_table, False

                cost = self.data.loc[j, 'Costs']
                # Skip if already evaluated
                if pd.notna(cost)[0] and not np.isclose(float(cost), -1, atol=1e-9):
                    HYGO_table.individuals[int(self.data.loc[j,'Individuals'])].cost = HYGO_params.badvalue
                    self.data.loc[j, (rep_name,'Cost_terms')] = np.nan
                    self.data.loc[j, (rep_name,'Evaluation_time')] = np.nan
                    self.data.loc[j, ('Uncertainty','Minimum')] = np.nan
                    self.data.loc[j, ('Uncertainty','All')] = np.nan
                    if HYGO_params.individual_paths:
                        ipath = os.path.join(rep_path, f"Individual{j}")
                        os.makedirs(ipath, exist_ok=True)
                        self.data.loc[j, (rep_name,'Path')] = ipath
                        HYGO_table.individuals[int(self.data.loc[j,'Individuals'])].path.append(ipath)
                    invalid_map[k] = j
                    if HYGO_params.security_backup:
                        self._backup_state(HYGO_table, base_path)
                else:
                    valid_idxs.append(j)
                    valid_params.append(batch_inds[k].parameters)
                    if HYGO_params.individual_paths:
                        ipath = os.path.join(rep_path, f"Individual{j}")
                        os.makedirs(ipath, exist_ok=True)
                        valid_paths.append(ipath)
                    else:
                        valid_paths.append(None)

            # Evaluate cost function only if we have valid individuals
            if not valid_params:
                continue
            if HYGO_params.verbose:
                if HYGO_params.batch_evaluation:
                    print(f'\t Evaluating individuals {self.data.loc[batch_idxs,"Individuals"].astype(int).tolist()}')
                else:
                    print(f'\t Evaluating individuals {int(self.data.loc[batch_idxs,"Individuals"].tolist()[0])}')
            cost_out = HYGO_params.cost_function(valid_params, valid_paths) if len(valid_params) > 1 else HYGO_params.cost_function(valid_params[0], valid_paths[0])
            J, J_vals = (cost_out if isinstance(cost_out, (list, tuple)) and len(cost_out) == 2 else ([], []))
            if len(valid_params) == 1:
                J, J_vals = [J], [J_vals]

            # Store evaluation results
            for pos, j in enumerate(valid_idxs):
                self.data.loc[j, (rep_name,'Cost')] = J[pos]
                self.data.loc[j, (rep_name,'Cost_terms')] = str(J_vals[pos])
                self.data.loc[j, (rep_name,'Evaluation_time')] = time.time() - ref_time
                if HYGO_params.individual_paths:
                    self.data.loc[j, (rep_name,'Path')] = valid_paths[pos]
                    HYGO_table.individuals[int(self.data.loc[j,'Individuals'])].path.append(valid_paths[pos])
                HYGO_table.individuals[int(self.data.loc[j,'Individuals'])].cost = J[pos]
                self.idx_to_check.append(j)

            a = 1
            for _ in range(actual_size):
                self.idx_to_evaluate[rep].pop(0)
            
            if HYGO_params.security_backup:
                    self._backup_state(HYGO_table, base_path)

        return HYGO_table, True
    
    def evaluate_population(self, idx_to_evaluate, HYGO_params, HYGO_table, path, exploitation=False):
        """
        Evaluate individuals in the population using the defined cost function.

        Parameters:
            idx_to_evaluate (list): List of indices per repetition.
            HYGO_params (object): Parameters for the Genetic Algorithm.
            HYGO_table (Table): Table storing individuals and metadata.
            path (str): Directory to store evaluation data.
            exploitation (bool): Whether the evaluation is part of exploitation.

        Returns:
            HYGO_table (Table): Updated table.
            checker (bool): Flag for evaluation convergence.
        """
        
        checker = True
        # self.idx_to_evaluate = idx_to_evaluate
        base_path = os.path.join(path, f'Gen{self.generation}')

        if HYGO_params.verbose and not exploitation:
            print(f"################ Evaluation of generation {self.generation} ################")

        if HYGO_params.individual_paths or HYGO_params.security_backup:
            os.makedirs(base_path, exist_ok=True)

        for rep in range(HYGO_params.repetitions):
            if HYGO_params.verbose:
                print(f'---------------- Repetition {rep+1} ----------------')
            HYGO_table, checker = self.evaluate_and_store(HYGO_params, HYGO_table, base_path, rep, copy.deepcopy(idx_to_evaluate[rep]), exploitation)

        self.idx_to_check = list(np.unique(self.idx_to_check))
        non_valid = self._handle_uncertainty(HYGO_params, HYGO_params.repetitions)

        if non_valid:
            if HYGO_params.repetitions > 1 and HYGO_params.repeat_indivs_outside_uncertainty:
                if HYGO_params.verbose:
                    print(f'---------------- Repetition {HYGO_params.repetitions+1} ----------------')
                    print(f'{len(non_valid)} individuals outside uncertainty')
                if len(self.idx_to_evaluate) <= HYGO_params.repetitions:
                    self.idx_to_evaluate.append(non_valid)
                else:
                    self.idx_to_evaluate[HYGO_params.repetitions] += non_valid
                    self.idx_to_evaluate[HYGO_params.repetitions] = np.unique(self.idx_to_evaluate[HYGO_params.repetitions]).tolist()
                    
                if not exploitation and ((f'Rep {HYGO_params.repetitions+1}', 'Cost') not in self.data.columns):
                    self.add_repetition()
                    
                HYGO_table, checker = self.evaluate_and_store(HYGO_params, HYGO_table, base_path, HYGO_params.repetitions, copy.deepcopy(non_valid), exploitation)
                non_valid = self._handle_uncertainty(HYGO_params, HYGO_params.repetitions+1)
                #Assign badvalue
                for idx in non_valid:
                    self.data.loc[idx, 'Costs'] = HYGO_params.badvalue
                if HYGO_params.verbose:
                    print(f'{len(non_valid)} individuals have been assigned a badvalue')
        # print(self.data[('Uncertainty', 'All')])
        if (HYGO_params.check_type == 'Neval' or HYGO_params.limit_evaluations) and not checker:
            to_drop = [i for i in range(self.data.shape[0]) if int(self.data.loc[i,'Individuals']) >= HYGO_params.neval]
            for i in to_drop:
                HYGO_table.remove_individual(-1)
            self.data = self.data.drop(to_drop).reset_index(drop=True)

        self.state = 'Evaluated'
        idx = self.data['Individuals']
        for i, j in enumerate(idx):
            HYGO_table.individuals[int(j)].cost = float(self.data.loc[i, 'Costs'])
            HYGO_table.costs[int(j)] = float(self.data.loc[i, 'Costs'])
        self.sort_pop()
        if HYGO_params.security_backup:
            self._backup_state(HYGO_table, base_path)
        return HYGO_table, checker

    def _backup_state(self, HYGO_table, path):
        try:
            import dill
            file = open(path+'/pop_backup.obj','wb')
            dill.dump(self,file)
            file.close()
            file = open(path+'/table_backup.obj','wb')
            dill.dump(HYGO_table,file)
            file.close()
        except ImportError:
            print("[Warning] dill not installed. Backup skipped.")

    def _handle_uncertainty(self, HYGO_params, nreps):
        non_valid_idx = []
        for idx in self.idx_to_check:
            minun, valid_cost, uncertainties = self.compute_uncertainty(idx, nreps)
            self.data.loc[idx, ('Uncertainty', 'Minimum')] = minun
            self.data.loc[idx, ('Uncertainty', 'All')] = str(uncertainties)
            if HYGO_params.repetitions > 1 and minun > HYGO_params.uncertainty:
                non_valid_idx.append(idx)
            else:
                self.data.loc[idx, 'Costs'] = valid_cost
        self.idx_to_check = []
        return non_valid_idx
    
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
        valid_cost = -1

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
