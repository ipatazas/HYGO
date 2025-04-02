import numpy as np
from ..individual import *
from scipy.optimize import curve_fit
import copy
import hygo
import os

def reconstruction_cost(HYGO_params,parameters,path=None):
    '''
    Dummy cost function, it yields the value of cost function given the parameters of an individual,
    it can also serve to measure any experiment since the parameters of each individual are passed
    and such individual has to be measured

    Must have 2 outputs:
        J(float): overall cost function value
        J_terms(list): all cost values from which the J is derived
    '''

    J = []
    objective = HYGO_params.objective_vals
    norm = np.zeros(HYGO_params.control_outputs)

    for ii in range(len(parameters)):
        results = np.zeros((len(parameters[ii]),HYGO_params.N_control_points))

        for i in range(HYGO_params.control_outputs):
            # Obtain a function
            law = parameters[ii][i]

            f = lambda t,s : eval(law)

            results[i,:] = f(HYGO_params.evaluation_time_sample,HYGO_params.ControlPoints)

            norm[i] = np.power(np.linalg.norm(np.array(objective[i])-np.array(results[i])),2)/np.linalg.norm(np.array(results[i]))
        
        J.append(float(np.mean(norm)))

    return J,[0]*len(J)

def regenerate_matrix(HYGO_params,ObjectiveControlPoints):
        '''
        This method generates an individual that fits as best as possible to the ObjectiveControlPoints
        by calling HYGO and obtaining aminimum to an objective function.
        
        Parameters:
            - HYGO_params(object):  An object containing genetic algorithm parameters.
            - ObjectiveControlPoints(np.array): Noutx4 matrix containing the objective control points
            
        Returns:
            - ind(object:Individual): an individual that has the best fit to the ObjectiveControlPoints
        '''
        
        # Copy the parameters of the problem
        reconstruction_params = copy.deepcopy(HYGO_params)
        # Update parameters for reconstruction
        reconstruction_params.name = 'ReconstructionProblem'
        reconstruction_params.verbose = False
        reconstruction_params.plotter = None
        reconstruction_params.pop_size = reconstruction_params.reconstruction_pop_size
        reconstruction_params.ngen = int(np.ceil(reconstruction_params.reconstruction_neval/reconstruction_params.pop_size))
        reconstruction_params.repetitions = 1
        reconstruction_params.uncertainty = 0
        reconstruction_params.individual_paths = False
        reconstruction_params.security_backup = False
        reconstruction_params.limit_evaluations = True
        reconstruction_params.neval = reconstruction_params.reconstruction_neval
        reconstruction_params.exploitation = False
        reconstruction_params.objective_vals = ObjectiveControlPoints
        reconstruction_params.check_convergence = True
        reconstruction_params.check_type = 'Generation_change'
        reconstruction_params.generations_stuck = int(np.ceil(0.4*reconstruction_params.ngen))
        reconstruction_params.batch_evaluation = True
        reconstruction_params.batch_size = reconstruction_params.pop_size
        func =  lambda reconstruction_params,parameters,path = None : reconstruction_cost(reconstruction_params,parameters,path)
        setattr(reconstruction_params,'cost_function',func.__get__(reconstruction_params, reconstruction_params.__class__))
        
        import time
        ref = time.time()
        #Create the object
        reconstruction = hygo.HYGO(reconstruction_params)
        #Run the genetic algorithm
        reconstruction.go()
        # Get the best individual index
        ind_idx,_ = reconstruction.table.give_best(1)
        # Get the best individual
        ind = copy.deepcopy(reconstruction.table.individuals[int(ind_idx)])
        ind.reconstruction_time = time.time()-ref
        if HYGO_params.verbose:
            print(f'Reconstruction time = {ind.reconstruction_time}')
        
        del reconstruction

        return ind

class Simplex:
    """
    The Simplex class represents the implementation of the Downhill Simplex optimization algorithm.

    Attributes:
        simplex_state (str): Current state of the simplex optimization process.
        exploitation (bool): Flag indicating whether exploitation is enabled.
        simplex_idx (list): List of indices representing individuals in the simplex.
        simplex_costs (list): List of costs associated with individuals in the simplex.
        simplex_centroid (list): List representing the centroid of the simplex.
        simplex_cycle (int): Current cycle or iteration of the simplex optimization.
        simplex_memory (list): List containing memory of past simplex configurations.
        simplex_cycle_checker (bool): Checks that the last Downhill simplex cycle has finalized.

    Methods:
        __init__: Initializes the Simplex object.
        initialize_simplex: Initializes the simplex for the Downhill Simplex method.
        exploitation_simplex: Performs the exploitation phase of the Downhill Simplex method.
        simplex_ordering: Orders the individuals in the simplex based on their costs.
        simplex_centroid_computation: Computes the centroid of the simplex.
        simplex_Reflection: Performs the Reflection step in the simplex optimization.
        simplex_single_Contraction: Performs a single Contraction step in the simplex optimization.
        simplex_shrink: Shrinks the simplex based on the best individual.
        simplex_expanse: Expands the simplex based on a reflected individual.
        simplex_batch_cyle: Computes a simplex cycle in a batch method
        check_convergence: Checks for convergence in the simplex optimization.
        fixation_checker: Checks if individuals in the simplex are stuck in a hyperplane.
        create_individual: Creates an individual and adds its preliminary information to the table and data
        func: Function that computes the hyperplane in n dimensions.
        R2: Computes the R-squared value for the hyperplane.
    """

    def __init__(self):
        """
        Initializes the Simplex object.
        """
        self.simplex_state = None
        pass

    def initialize_simplex(self,HYGO_params,HYGO_table):
        """
        Initializes the simplex for the Downhill Simplex method.

        Args:
            - HYGO_params (object): Object containing genetic algorithm parameters.
            - HYGO_table (object): The table containing information about individuals.

        Raises:
            ValueError: If the provided exploitation parameter is not a boolean or a list of booleans.

        Returns:
            None
        """

        if HYGO_params.verbose:
            print('---> Downhill Simplex method selected for generation '+str(self.generation))
            print('    -Initialization:')

        # Set the exploitation flag based on the provided parameter
        if type(HYGO_params.exploitation)==list:
            self.exploitation = HYGO_params.exploitation[self.generation-1]
        elif type(HYGO_params.exploitation)==bool:
            self.exploitation = HYGO_params.exploitation
        else:
            raise ValueError('The exploitation must be a bool or a list of bools')
        
        # Initialize simplex-related attributes
        self.simplex_idx = (np.zeros(HYGO_params.SimplexSize)-1).tolist()
        self.simplex_costs = (np.zeros(HYGO_params.SimplexSize)-1).tolist()
        
        if HYGO_params.optimization == 'Parametric':
            self.simplex_centroid = (np.zeros(HYGO_params.N_params)-1).tolist()
        elif HYGO_params.optimization == 'Control Law' and not HYGO_params.SimplexInterpolation:
            self.simplex_centroid = (np.zeros((HYGO_params.control_outputs,HYGO_params.N_control_points))-1).tolist()
        
        self.simplex_cycle = 1
        self.simplex_memory = []
        self.simplex_cycle_checker = True

        # Handle different simplex pool options
        if HYGO_params.SimplexPool == 'Population':
            # Only individuals from the current population will be taken into account for building the simplex
            if HYGO_params.SimplexInitialization=='BestN':
                # Set the values for the indexes and costs if the best individuals are to be taken
                self.simplex_idx = np.array(self.data.loc[0:HYGO_params.SimplexSize-1,'Individuals'])
                self.simplex_costs = np.array(self.data.loc[0:HYGO_params.SimplexSize-1,'Costs'])
                
            elif HYGO_params.SimplexInitialization=='ClosestN':
                # Perform the kcloses calculation to initialize the simplex
                from .findKClosest import findKClosest
                
                # Take the data of the best individual
                best_idx = int(self.data.loc[0,'Individuals'])
                
                if HYGO_params.optimization == 'Parametric':
                    best_parameters = HYGO_table.individuals[best_idx].parameters
                elif HYGO_params.optimization == 'Control Law':
                    best_parameters = HYGO_table.individuals[best_idx].ControlPoints
                    
                best_cost = float(self.data.loc[0,'Costs'])
                
                # Obtain the resto of the individuals' data
                idx = np.array(self.data.loc[1:,'Individuals'])
                params = []
                costs = []
                for i in idx:
                    if HYGO_params.optimization == 'Parametric':
                        params.append(HYGO_table.individuals[int(i)].parameters)
                    elif HYGO_params.optimization == 'Control Law':
                        params.append(HYGO_table.individuals[int(i)].ControlPoints)
                    costs.append(HYGO_table.individuals[int(i)].cost)
                costs=np.array(costs)
                
                # Perform the k-closest computation
                new_idx = findKClosest(params,best_parameters,HYGO_params.SimplexSize-1)
                costs = costs[new_idx]
                
                # Save the data
                self.simplex_idx = np.array([best_idx]+idx[new_idx.tolist()])
                self.simplex_costs = np.array([best_cost]+costs.tolist())
            else:
                raise ValueError('SimplexInitialization not valid')
        elif HYGO_params.SimplexPool == 'All':
            # All individuals  will be taken into account for building the simplex
            if HYGO_params.SimplexInitialization=='BestN':
                # Get the best N individuals
                self.simplex_idx,self.simplex_costs = HYGO_table.give_best(HYGO_params.SimplexSize)
                
            elif HYGO_params.SimplexInitialization=='ClosestN':
                from .findKClosest import findKClosest
                
                # Obtain the best individual data
                best_idx,best_cost = HYGO_table.give_best(1)
                best_idx = int(best_idx)
                best_cost=float(best_cost)
                
                # Obtain the resto of the individuals' data
                if HYGO_params.optimization == 'Parametric':
                    best_parameters = HYGO_table.individuals[best_idx].parameters
                elif HYGO_params.optimization == 'Control Law':
                    best_parameters = HYGO_table.individuals[best_idx].ControlPoints
                idx = [i for i in range(len(HYGO_table.individuals)) if i!=best_idx]
                params = []
                costs = []
                for i in idx:
                    if HYGO_params.optimization == 'Parametric':
                        params.append(HYGO_table.individuals[int(i)].parameters)
                    elif HYGO_params.optimization == 'Control Law':
                        params.append(HYGO_table.individuals[int(i)].ControlPoints)
                    costs.append(HYGO_table.individuals[int(i)].cost)
                costs=np.array(costs)
                
                # Perform the k-closest computation
                new_idx = findKClosest(params,best_parameters,HYGO_params.SimplexSize-1)
                costs = costs[new_idx]
                
                # Save the data
                self.simplex_idx = np.array([best_idx]+idx[new_idx.tolist()])
                self.simplex_costs = np.array([best_cost]+costs.tolist())
            else:
                raise ValueError('SimplexInitialization not valid')
        else:
            raise ValueError('The Simplex pool has to either be Population or All')

        # Update the state
        self.simplex_state = 'Exploitation initialized'

        if HYGO_params.verbose:
            print('     --->DONE')

    def exploitation_simplex(self,HYGO_params,HYGO_table,path):
        """
        Executes the exploitation phase using the Downhill Simplex method.

        Args:
            HYGO_params (object): An object containing genetic algorithm parameters.
            HYGO_table (object): The  table containing information about individuals.
            path (str): The path where the results are stored.

        Returns:
            Tuple[PopulationTable, bool, bool]: A tuple containing the updated population table, a flag indicating whether the maximum number of simplex cycles is reached, and a flag indicating convergence.
        """
        
        # Obtain the total number of individuals that the population should have after the simplex is completed
        pop_size = HYGO_params.pop_size + HYGO_params.SimplexOffspring
        
        # Obtain the current number of individuals in the population
        nind = self.data.shape[0]

        # Initialize the flags
        checker = True
        counter = 0
        convergence = False

        # Counter used to know when the fixation checker must be applied
        if self.simplex_cycle < HYGO_params.SimplexCycleChecker:
            idx_check = HYGO_params.SimplexCycleChecker
        else:
            idx_check = self.simplex_cycle

        # Loop until number of generated individuals reaches "pop_size"
        while nind<pop_size and checker and self.simplex_cycle<HYGO_params.MaxSimplexCycles and not convergence:
            
            # Check that the last cycle finished computation, used for the security backup
            if not self.simplex_cycle_checker:
                if HYGO_params.verbose:
                    print('----- Previous cycle not finished -----')
                
                self.simplex_idx = self.simplex_memory[-1]
                for i,idx in enumerate(self.simplex_idx):
                    self.simplex_costs[i] = HYGO_table.individuals[int(idx)].cost
            else:
                # Select option for memory: "Population" or "Database" and update the simplex indexes and costs
                if HYGO_params.SimplexPool == 'Population':
                    self.simplex_idx = np.array(self.data.loc[0:HYGO_params.SimplexSize-1,'Individuals'])
                    self.simplex_costs = np.array(self.data.loc[0:HYGO_params.SimplexSize-1,'Costs'])
                elif HYGO_params.SimplexPool == 'All':
                    self.simplex_idx,self.simplex_costs = HYGO_table.give_best(HYGO_params.SimplexSize)
            
            if HYGO_params.verbose:
                print('******** Cycle '+str(self.simplex_cycle)+' ********')
                print('    -Ordering')
                
            # Order simplex
            self.simplex_ordering()
            if HYGO_params.verbose:
                print('    -Centroid')
                
            # Compute the centroid
            if (HYGO_params.optimization == 'Parametric') or (HYGO_params.optimization == 'Control Law' and not HYGO_params.SimplexInterpolation):
                self.simplex_centroid_computation(HYGO_params,HYGO_table)
            
            # Add memory if the previous cycle finished
            if self.simplex_cycle_checker:
                self.simplex_memory.append(self.simplex_idx)
            
            # Check fixation
            if self.simplex_cycle==idx_check and HYGO_params.optimization == 'Parametric':
                HYGO_table,checker,idx_inserted = self.fixation_checker(self.simplex_memory[-HYGO_params.SimplexCycleChecker:],
                                                         HYGO_params,HYGO_table,path,checker)
                if idx_inserted:
                    idx_check += HYGO_params.SimplexCycleChecker
                else:
                    idx_check +=1
            
            # Start the new cycle computation
            self.simplex_cycle_checker = False
            
            # Simplex procedure
            if hasattr(HYGO_params,'batch_evaluation') and hasattr(HYGO_params,'batch_size') and HYGO_params.batch_evaluation:
                HYGO_table,checker = self.simplex_batch_cyle(HYGO_params,HYGO_table,path)
            else:
                HYGO_table,checker = self.simplex_Reflection(HYGO_params,HYGO_table,path)
            
            # Indicate that the cycle finished
            self.simplex_cycle_checker = True

            # Update the cycle counter
            self.simplex_cycle+=1

            # Update the number of individuals in the population
            nind = self.data.shape[0]
            self.Nind = nind
            
            counter+=1
            
            # Check convergence
            if HYGO_params.check_convergence:
                convergence = self.check_convergence(HYGO_params,HYGO_table)
                
            # Make a security backup if specified
            if HYGO_params.security_backup:
                import dill
                file = open(path+'/Gen'+str(self.generation)+'/pop_backup.obj','wb')
                dill.dump(self,file)
                file.close()
                file = open(path+'/Gen'+str(self.generation)+'/table_backup.obj','wb')
                dill.dump(HYGO_table,file)
                file.close()

        self.simplex_state = 'Simplex done'
        
        if HYGO_params.security_backup:
            import dill
            file = open(path+'/Gen'+str(self.generation)+'/pop_backup.obj','wb')
            dill.dump(self,file)
            file.close()
            file = open(path+'/Gen'+str(self.generation)+'/table_backup.obj','wb')
            dill.dump(HYGO_table,file)
            file.close()

        return HYGO_table,checker,convergence

    def simplex_ordering(self):
        """
        Orders the individuals in the simplex based on their costs in ascending order.

        The function uses the numpy.argsort function to obtain the indices that would sort the costs in ascending order.
        It then rearranges the simplex indices and costs arrays accordingly.

        Note: This function modifies the simplex_idx and simplex_costs attributes in-place.

        Returns:
            None
        """

        # Obtain indexes and costs
        idx = np.array(self.simplex_idx)
        costs = np.array(self.simplex_costs)

        # Order the costs in ascending order and obtain the positions
        ordered = np.argsort(costs)

        # Obtain the indexes and costs ordered
        order_idx = idx[ordered].tolist()
        order_costs = costs[ordered].tolist()

        # Update the simplex attributes
        self.simplex_idx = order_idx
        self.simplex_costs = order_costs

    def simplex_centroid_computation(self,HYGO_params,HYGO_table):
        """
        Computes the centroid of the valid individuals in the simplex.

        The centroid is computed as the mean of the parameter values of the valid individuals. 
        Valid individuals are those with costs lower than HYGO_params.badvalue/1000.

        Args:
            HYGO_params (object): An object containing parameters for the genetic algorithm.
            HYGO_table (object): Table containing the population of individuals.

        Raises:
            ValueError: If no valid solutions are found in order to build the simplex.

        Returns:
            None
        """
        # Identify valid individuals in the simplex based on cost
        good_indiv = np.array(self.simplex_costs)<HYGO_params.badvalue/1000
        idx = np.array(self.simplex_idx)
        valid_idx = idx[good_indiv]

        # Raise an error if no valid solutions are found
        if np.sum(valid_idx)==0:
            raise ValueError('No valid solutions were found in order to build the simplex')

        # Allocate memory for parameter values of valid individuals
        if np.sum(good_indiv)>1:
            if HYGO_params.optimization == 'Parametric':
                values = np.zeros((valid_idx.size-1,HYGO_params.N_params))
                # Extract parameters. (N-1) best individuals
                for i,idx in enumerate(valid_idx[0:-1]):
                    params = HYGO_table.individuals[int(idx)].parameters
                    values[i,:] = params
                # Compute mean value to obtain the simplex centroid
                self.simplex_centroid = np.mean(values,axis=0).tolist()
            
            elif HYGO_params.optimization == 'Control Law':
                for law_idx in range(HYGO_params.control_outputs):
                    values = np.zeros((valid_idx.size-1,HYGO_params.N_control_points))
                    # Extract parameters. (N-1) best individuals
                    for i,idx in enumerate(valid_idx[0:-1]):
                        params = HYGO_table.individuals[int(idx)].ControlPoints
                        values[i,:] = params
                    self.simplex_centroid[law_idx] = np.mean(values,axis=0).tolist()
                    
        elif np.sum(good_indiv)==1:
            # If only one valid individual, set the centroid to its parameters
            if HYGO_params.optimization == 'Parametric':
                self.simplex_centroid = HYGO_table.individuals[int(valid_idx)].parameters
            elif HYGO_params.optimization == 'Control Law':
                self.simplex_centroid = HYGO_table.individuals[int(valid_idx)].ControlPoints
                
        # Check for any NaN values in the computed centroid
        if np.sum(np.isnan(self.simplex_centroid))>0:
            raise ValueError('No valid solutions were found in order to build the simplex')

    def simplex_Reflection(self,HYGO_params,HYGO_table,path):
        """
        Applies the Reflection operation to the simplex.

        The Reflection operation involves reflecting the worst individual across the centroid of the valid individuals 
        in the simplex. The new individual is then added to the population, and further operations may be applied based 
        on the comparison of its cost to other individuals in the simplex.

        Args:
            HYGO_params (object): An object containing parameters for the genetic algorithm.
            HYGO_table (object): Table containing the population of individuals.
            path (str): The path to save additional information about the operations performed.

        Returns:
            tuple: A tuple containing the updated HYGO_table and a boolean value (checker) indicating whether the maximum number 
            of individuals was reached.
        """
        # Get costs of the worst and second-to-last individuals in the simplex
        J1 = self.simplex_costs[0]
        J_end_minus_1 = self.simplex_costs[-2]
        
        # Identify the worst individual
        ind_end = HYGO_table.individuals[int(self.simplex_idx[-1])]

        if HYGO_params.verbose:
            print('    -Reflection:')
        
        # Compute centroid
        if (HYGO_params.optimization == 'Parametric') or (HYGO_params.optimization == 'Control Law' and not HYGO_params.SimplexInterpolation):
            centroid = np.array(self.simplex_centroid)

        if HYGO_params.optimization == 'Parametric':
            # Get parameters of the worst individual and the centroid of the simplex
            to_reflect = np.array(ind_end.parameters)
            # Compute the reflected parameters
            reflected = centroid + HYGO_params.reflection_alpha*(centroid-to_reflect)

            # Ensure the new parameters are inside the bounds
            reflected = self.check_params(reflected.tolist(),HYGO_params)

            # Create a new individual with the reflected parameters
            new_ind = Individual()
            new_ind.create(HYGO_params=HYGO_params,params=reflected)

            # Add the new individual to the table
            [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_ind)

            # Assume that the individual is valid
            valid = True

            # Check if there is a custom validity function
            if hasattr(HYGO_params,'validity') and callable(HYGO_params.validity):
                custom_valid = HYGO_params.validity(new_ind.parameters)
                valid = valid and custom_valid

            # Get the index of the individual in the population
            nind = self.data.shape[0]
            
            # Add the individual to the population and table data if it does not exist and if it is valid
            if not exists:
                for rep in range(HYGO_params.repetitions):
                    self.idx_to_evaluate[rep] = self.idx_to_evaluate[rep] + [nind]
                    
                # Update information in the data structure
                self.data.loc[nind,'Individuals'] = idx
                self.data.loc[nind,('Parents','first')]  = int(self.simplex_idx[-1])
                self.data.loc[nind,('Parents','second')] = -1
                self.data.loc[nind,('Operation','Type')] = 'Reflection'
                self.data.loc[nind,('Operation','Point_parts')]  = 'None'
                
                if valid:
                    # Evaluate the individual and update the HYGO_table
                    HYGO_table,checker = self.evaluate_population(copy.deepcopy(self.idx_to_evaluate),HYGO_params,HYGO_table,path,True)
                else:
                    if HYGO_params.verbose:
                        print('Individual not evaluated, it was labeled as not valid')
                    self.data.loc[nind,'Costs'] = HYGO_params.badvalue
                    HYGO_table.individuals[idx].cost = HYGO_params.badvalue
                    if (HYGO_params.check_type=='Neval' or HYGO_params.limit_evaluations) and len(HYGO_table.individuals)>=HYGO_params.neval:
                        checker = False
                    else:
                        checker = True

            else:
                checker = True

        elif HYGO_params.optimization == 'Control Law':
            if HYGO_params.SimplexInterpolation:
                #Obtain the law
                law,new_parents,new_coeff = self.interpolate_simplex(HYGO_params,HYGO_table,'Reflection')
                
                # Create the new individual
                new_ind = Individual()
                new_ind.parameters = law
                new_ind.evaluate_ControlPoints(HYGO_params)
                
                checker = True # Flag to check if a valid mutation is found
                counter = 1 # Counter for limiting the number of attempts to find a valid mutation
                while checker:
                    # re-generate the matrix
                    new_ind = regenerate_matrix(HYGO_params,new_ind.ControlPoints)
                    # Store the parents and soefficients
                    new_ind.simplex_parents = new_parents
                    new_ind.coefficients = new_coeff
                    # Add the new individual to the table
                    [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_ind)

                    checker = HYGO_params.remove_duplicates and counter<HYGO_params.MaxTries and exists
                    counter +=1
                    
            else:
                # Get parameters of the worst individual and the centroid of the simplex
                to_reflect = np.array(ind_end.ControlPoints)
                # Compute the reflected parameters
                reflected = 2*np.array(centroid) - to_reflect
                # Create the new individual
                new_ind = Individual()
                
                checker = True # Flag to check if a valid mutation is found
                counter = 1 # Counter for limiting the number of attempts to find a valid mutation
                while checker:
                    # re-generate the matrix
                    new_ind = regenerate_matrix(HYGO_params,reflected)
                    # Add the new individual to the table
                    [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_ind)
                    
                    checker = HYGO_params.remove_duplicates and counter<HYGO_params.MaxTries and exists
                    counter +=1
                
            # Get the index of the individual in the population
            nind = self.data.shape[0]
            
            # Check if the control is within bounds
            valid = [0]*HYGO_params.control_outputs
            for i in range(HYGO_params.control_outputs):
                valid[i] = int(np.sum(new_ind.ControlPoints[i,:]<np.array(HYGO_params.Control_range[i][0])) + np.sum(new_ind.ControlPoints[i,:]>np.array(HYGO_params.Control_range[i][1]))) == 0
            valid = int(np.sum(np.array(valid))) == len(valid)

            # Check if there is a custom validity function
            if hasattr(HYGO_params,'validity') and callable(HYGO_params.validity):
                custom_valid = HYGO_params.validity(new_ind.parameters)
                valid = valid and custom_valid
            
            # Add the individual to the population and table data if it does not exist
            if not exists:
                for rep in range(HYGO_params.repetitions):
                    self.idx_to_evaluate[rep] = self.idx_to_evaluate[rep] + [nind]
                    
                # Update information in the data structure
                self.data.loc[nind,'Individuals'] = idx
                self.data.loc[nind,('Parents','first')]  = int(self.simplex_idx[-1])
                self.data.loc[nind,('Parents','second')] = -1
                self.data.loc[nind,('Operation','Type')] = 'Reflection'
                self.data.loc[nind,('Operation','Point_parts')]  = 'None'
                
                if valid:
                    # Evaluate the individual and update the HYGO_table
                    HYGO_table,checker = self.evaluate_population(copy.deepcopy(self.idx_to_evaluate),HYGO_params,HYGO_table,path,True)
                else:
                    if HYGO_params.verbose:
                        print('Individual not evaluated, it was labeled as not valid')
                    self.data.loc[nind,'Costs'] = HYGO_params.badvalue
                    HYGO_table.individuals[idx].cost = HYGO_params.badvalue
                    if (HYGO_params.check_type=='Neval' or HYGO_params.limit_evaluations) and len(HYGO_table.individuals)>=HYGO_params.neval:
                        checker = False
                    else:
                        checker=True
            else:
                checker = True
                    
        if checker:
            # Get the cost of the reflected individual
            Jr = HYGO_table.individuals[idx].cost

        # Check the cost of the reflected individual and perform additional operations if necessary
        if checker:
            self.Nind+=1
            if np.array(Jr) >= np.array(J_end_minus_1):
                HYGO_table,checker = self.simplex_single_Contraction(HYGO_params,HYGO_table,path,int(idx))
            elif np.array(Jr) < np.array(J1):
                HYGO_table,checker = self.simplex_expanse(HYGO_params,HYGO_table,path,int(idx))

        return HYGO_table,checker

    def simplex_single_Contraction(self,HYGO_params,HYGO_table,path,Reflection_idx):
        """
        Applies a single Contraction operation to the simplex.

        The single Contraction operation involves contracting the simplex towards the best of the reflected 
        individual and the worst individual. The new individual is then added to the population, and further 
        operations may be applied based on the comparison of its cost to other individuals in the simplex.

        Args:
            HYGO_params (object): An object containing parameters for the genetic algorithm.
            HYGO_table (object): Table containing the population of individuals.
            path (str): The path to save additional information about the operations performed.
            Reflection_idx (int): Index of the individual in the population that was reflected.

        Returns:
            tuple: A tuple containing the updated HYGO_table and a boolean value (checker) indicating whether the maximum number 
            of individuals was reached.
        """
        # Get the worst individual's index and cost
        ind_end = HYGO_table.individuals[int(self.simplex_idx[-1])]
        J_end = self.simplex_costs[-1]

        # Get the reflected individual
        ind_reflect = HYGO_table.individuals[Reflection_idx]

        if HYGO_params.verbose:
                print('    -Single Contraction:')
        
        # Compute centroid
        if (HYGO_params.optimization == 'Parametric') or (HYGO_params.optimization == 'Control Law' and not HYGO_params.SimplexInterpolation):
            centroid = np.array(self.simplex_centroid)
        
        if HYGO_params.optimization == 'Parametric':
            # Choose parameters for Contraction based on the comparison of costs
            if ind_reflect.cost<J_end:
                to_contract = np.array(ind_reflect.parameters)
            else:
                to_contract = np.array(ind_end.parameters)

            # Compute the contracted parameters
            contracted = centroid + HYGO_params.contraction_rho*(to_contract-centroid) 
            
            # Ensure the new parameters are inside the bounds
            contracted = self.check_params(contracted.tolist(),HYGO_params)

            # Create a new individual with the contracted parameters
            new_ind = Individual()
            new_ind.create(HYGO_params=HYGO_params,params=contracted)

            # Add the new individual to the table
            [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_ind)

            # Assume that the individual is valid
            valid = True

            # Check if there is a custom validity function
            if hasattr(HYGO_params,'validity') and callable(HYGO_params.validity):
                custom_valid = HYGO_params.validity(new_ind.parameters)
                valid = valid and custom_valid

            # Get the index of the individual in the population
            nind = self.data.shape[0]

            # Add the individual to the population and table data if it does not exist
            if not exists:
                for rep in range(HYGO_params.repetitions):
                        self.idx_to_evaluate[rep] = self.idx_to_evaluate[rep] + [nind]
                        
                # Update information in the data structure
                self.data.loc[nind,'Individuals'] = idx
                self.data.loc[nind,('Parents','first')]  = int(self.simplex_idx[-1])
                self.data.loc[nind,('Parents','second')] = -1
                self.data.loc[nind,('Operation','Type')] = 'Contraction'
                self.data.loc[nind,('Operation','Point_parts')]  = 'None'

                if valid:
                    # Evaluate the individual and update the HYGO_table
                    HYGO_table,checker = self.evaluate_population(copy.deepcopy(self.idx_to_evaluate),HYGO_params,HYGO_table,path,True)
                else:
                    if HYGO_params.verbose:
                        print('Individual not evaluated, it was labeled as not valid')
                    self.data.loc[nind,'Costs'] = HYGO_params.badvalue
                    HYGO_table.individuals[idx].cost = HYGO_params.badvalue
                    if (HYGO_params.check_type=='Neval' or HYGO_params.limit_evaluations) and len(HYGO_table.individuals)>=HYGO_params.neval:
                        checker = False
                    else:
                        checker = True
            else:
                checker = True

        elif HYGO_params.optimization == 'Control Law':
            if HYGO_params.SimplexInterpolation:#Perform interpolation
                # Obtain the law
                law,new_parents,new_coeff = self.interpolate_simplex(HYGO_params,HYGO_table,'Contraction')
                
                # Create the new individual
                new_ind = Individual()
                new_ind.parameters = law
                new_ind.evaluate_ControlPoints(HYGO_params)
                
                checker = True # Flag to check if a valid mutation is found
                counter = 1 # Counter for limiting the number of attempts to find a valid mutation
                while checker:
                    # re-generate the matrix
                    new_ind = regenerate_matrix(HYGO_params,new_ind.ControlPoints)
                    # Store the parents and soefficients
                    new_ind.simplex_parents = new_parents
                    new_ind.coefficients = new_coeff
                    # Add the new individual to the table
                    [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_ind)
                    
                    checker = HYGO_params.remove_duplicates and counter<HYGO_params.MaxTries and exists
                    counter +=1
                    
            else:
                # Choose parameters for Contraction based on the comparison of costs
                if ind_reflect.cost<J_end:
                    to_contract = np.array(ind_reflect.ControlPoints)
                else:
                    to_contract = np.array(ind_end.ControlPoints)
                
                # Compute the contracted parameters
                contracted = 0.5*(np.array(centroid) + to_contract)
                # Create the new individual
                new_ind = Individual()
                
                checker = True # Flag to check if a valid mutation is found
                counter = 1 # Counter for limiting the number of attempts to find a valid mutation
                while checker:
                    # re-generate the matrix
                    new_ind = regenerate_matrix(HYGO_params,contracted)
                    # Add the new individual to the table
                    [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_ind)
                    
                    checker = HYGO_params.remove_duplicates and counter<HYGO_params.MaxTries and exists
                    counter +=1
            
            # Get the index of the individual in the population
            nind = self.data.shape[0]

            # Check if the control is within bounds
            valid = [0]*HYGO_params.control_outputs
            for i in range(HYGO_params.control_outputs):
                valid[i] = int(np.sum(new_ind.ControlPoints[i,:]<np.array(HYGO_params.Control_range[i][0])) + np.sum(new_ind.ControlPoints[i,:]>np.array(HYGO_params.Control_range[i][1]))) == 0
            valid = int(np.sum(np.array(valid))) == len(valid)

            # Check if there is a custom validity function
            if hasattr(HYGO_params,'validity') and callable(HYGO_params.validity):
                custom_valid = HYGO_params.validity(new_ind.parameters)
                valid = valid and custom_valid

            # Add the individual to the population and table data if it does not exist
            if not exists:
                for rep in range(HYGO_params.repetitions):
                    self.idx_to_evaluate[rep] = self.idx_to_evaluate[rep] + [nind]
                    
                # Update information in the data structure
                self.data.loc[nind,'Individuals'] = idx
                self.data.loc[nind,('Parents','first')]  = int(self.simplex_idx[-1])
                self.data.loc[nind,('Parents','second')] = -1
                self.data.loc[nind,('Operation','Type')] = 'Contraction'
                self.data.loc[nind,('Operation','Point_parts')]  = 'None'
                
                if valid:
                    # Evaluate the individual and update the table
                    HYGO_table,checker = self.evaluate_population(copy.deepcopy(self.idx_to_evaluate),HYGO_params,HYGO_table,path,True)
                else:
                    if HYGO_params.verbose:
                        print('Individual not evaluated, it was labeled as not valid')
                    self.data.loc[nind,'Costs'] = HYGO_params.badvalue 
                    HYGO_table.individuals[idx].cost = HYGO_params.badvalue
                    if (HYGO_params.check_type=='Neval' or HYGO_params.limit_evaluations) and len(HYGO_table.individuals)>=HYGO_params.neval:
                        checker = False
                    else:
                        checker=True
            else:
                checker = True
                
        if checker:
            # Get the cost of the contracted individual
            Jc = HYGO_table.individuals[idx].cost

        # Check the cost of the contracted individual and perform additional operations if necessary
        if checker and np.array(Jc)>=np.array(J_end):
            self.Nind+=1
            HYGO_table,checker = self.simplex_shrink(HYGO_params,HYGO_table,path)

        return HYGO_table,checker

    def simplex_shrink(self,HYGO_params,HYGO_table,path):
        """
        Applies the shrink operation to the simplex.

        The shrink operation involves shrinking the simplex towards the best individual, creating new individuals
        with the shrunk parameters, and adding them to the population. The operation is performed for each individual
        in the simplex except the best individual.

        Args:
            HYGO_params (object): An object containing parameters for the genetic algorithm.
            HYGO_table (object): Table containing the population of individuals.
            path (str): The path to save additional information about the operations performed.

        Returns:
            tuple: A tuple containing the updated HYGO_table and a boolean value (checker) indicating whether the maximum
            number of individuals was reached
        """
        if HYGO_params.verbose:
            print('    -Shrink:')
        
        # Get the parameters of the best individual
        idx1 = self.simplex_idx[0]
        
        if HYGO_params.optimization == 'Parametric':
            params1 = HYGO_table.individuals[int(idx1)].parameters

            # Get the indices of individuals to shrink
            ids = self.simplex_idx[1:]

            shrinked = []

            # Perform the shrink operation for each individual in the simplex (except the best individual)
            for id in ids:
                params = HYGO_table.individuals[int(id)].parameters
                shrink = np.array(params1) + HYGO_params.shrinkage_sigma*(np.array(params)-np.array(params1)) #OLD
                shrink = self.check_params(shrink.tolist(),HYGO_params)
                shrinked.append(shrink)

            for i in range(len(ids)):
                # Create a new individual with the shrunk parameters
                new_ind = Individual()
                new_ind.create(HYGO_params=HYGO_params,params=shrinked[i])
                
                # Add the new individual to the table
                [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_ind)

                # Assume that the individual is valid
                valid = True

                # Check if there is a custom validity function
                if hasattr(HYGO_params,'validity') and callable(HYGO_params.validity):
                    custom_valid = HYGO_params.validity(new_ind.parameters)
                    valid = valid and custom_valid

                #Get index of individual in pop
                nind = self.data.shape[0]

                # Update the information of the individuals in the population
                if not exists:
                    for rep in range(HYGO_params.repetitions):
                        self.idx_to_evaluate[rep] = self.idx_to_evaluate[rep] + [nind]
                        
                    # Update information in the data structure
                    self.data.loc[nind,'Individuals'] = idx
                    self.data.loc[nind,('Parents','first')]  = int(ids[i])
                    self.data.loc[nind,('Parents','second')] = int(idx1)
                    self.data.loc[nind,('Operation','Type')] = 'Shrinked'
                    self.data.loc[nind,('Operation','Point_parts')]  = 'None'

                    if valid:
                        # Evaluate the individual and update the HYGO_table
                        HYGO_table,checker = self.evaluate_population(copy.deepcopy(self.idx_to_evaluate),HYGO_params,HYGO_table,path,True)
                    else:
                        if HYGO_params.verbose:
                            print('Individual not evaluated, it was labeled as not valid')
                        self.data.loc[nind,'Costs'] = HYGO_params.badvalue
                        HYGO_table.individuals[idx].cost = HYGO_params.badvalue
                        if (HYGO_params.check_type=='Neval' or HYGO_params.limit_evaluations) and len(HYGO_table.individuals)>=HYGO_params.neval:
                            checker = False
                        else:
                            checker = True
                else:
                    checker = True
                    
                if not checker:
                    break

        elif HYGO_params.optimization == 'Control Law':
            if HYGO_params.SimplexInterpolation:#Perform interpolation
                # Get the indices of individuals to shrink
                ids = self.simplex_idx[1:]
                # Perform the shrink operation for each individual in the simplex (except the best individual)
                for i in range(HYGO_params.SimplexSize-1):
                    # Obtain the law
                    law,new_parents,new_coeff = self.interpolate_simplex(HYGO_params,HYGO_table,'Shrinked',i+1)
                    
                    # Create the new individual
                    new_ind = Individual()
                    new_ind.parameters = law
                    new_ind.evaluate_ControlPoints(HYGO_params)
                    
                    checker = True # Flag to check if a valid mutation is found
                    counter = 1 # Counter for limiting the number of attempts to find a valid mutation
                    while checker:
                        # re-generate the matrix
                        new_ind = regenerate_matrix(HYGO_params,new_ind.ControlPoints)
                        # Store the parents and soefficients
                        new_ind.simplex_parents = new_parents
                        new_ind.coefficients = new_coeff
                        # Add the new individual to the table
                        [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_ind)
                        
                        checker = HYGO_params.remove_duplicates and counter<HYGO_params.MaxTries and exists
                        counter +=1
                        
                    # Get the index of the individual in the population
                    nind = self.data.shape[0]

                    # Check if the control is within bounds
                    valid = [0]*HYGO_params.control_outputs
                    for j in range(HYGO_params.control_outputs):
                        valid[j] = int(np.sum(new_ind.ControlPoints[j,:]<np.array(HYGO_params.Control_range[j][0])) + np.sum(new_ind.ControlPoints[j,:]>np.array(HYGO_params.Control_range[j][1]))) == 0
                    valid = int(np.sum(np.array(valid))) == len(valid)

                    # Check if there is a custom validity function
                    if hasattr(HYGO_params,'validity') and callable(HYGO_params.validity):
                        custom_valid = HYGO_params.validity(new_ind.parameters)
                        valid = valid and custom_valid

                    # Update the information of the individuals in the population
                    if not exists:
                        for rep in range(HYGO_params.repetitions):
                            self.idx_to_evaluate[rep] = self.idx_to_evaluate[rep] + [nind]
                            
                        # Update information in the data structure
                        self.data.loc[nind,'Individuals'] = idx
                        self.data.loc[nind,('Parents','first')]  = int(ids[i])
                        self.data.loc[nind,('Parents','second')] = int(idx1)
                        self.data.loc[nind,('Operation','Type')] = 'Shrinked'
                        self.data.loc[nind,('Operation','Point_parts')]  = 'None'
                        if valid:
                            # Evaluate the individual and update the table
                            HYGO_table,checker = self.evaluate_population(copy.deepcopy(self.idx_to_evaluate),HYGO_params,HYGO_table,path,True)
                        else:
                            if HYGO_params.verbose:
                                print('Individual not evaluated, it was labeled as not valid')
                            self.data.loc[nind,'Costs'] = HYGO_params.badvalue
                            HYGO_table.individuals[idx].cost = HYGO_params.badvalue
                            if (HYGO_params.check_type=='Neval' or HYGO_params.limit_evaluations) and len(HYGO_table.individuals)>=HYGO_params.neval:
                                checker = False
                            else:
                                checker=True
                    else:
                        checker = True
                        
                    if not checker:
                        break
                    
            else:
                params1 = HYGO_table.individuals[int(idx1)].ControlPoints
                
                # Get the indices of individuals to shrink
                ids = self.simplex_idx[1:]

                # Perform the shrink operation for each individual in the simplex (except the best individual)
                for id in ids:
                    # Obtain the individual to shrink Contro points
                    params = HYGO_table.individuals[int(id)].ControlPoints
                    # Compute the shrink operation
                    shrink = 0.5*(np.array(params)+np.array(params1))
                    
                    # Create the new individual
                    new_ind = Individual()
                    
                    checker = True # Flag to check if a valid mutation is found
                    counter = 1 # Counter for limiting the number of attempts to find a valid mutation
                    while checker:
                        # re-generate the matrix
                        new_ind = regenerate_matrix(HYGO_params,shrink)
                        # Add the new individual to the table
                        [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_ind)
                        
                        checker = HYGO_params.remove_duplicates and counter<HYGO_params.MaxTries and exists
                        counter +=1
                    
                    # Get the index of the individual in the population
                    nind = self.data.shape[0]
                
                    # Check if the control is within bounds
                    valid = [0]*HYGO_params.control_outputs
                    for i in range(HYGO_params.control_outputs):
                        valid[i] = int(np.sum(new_ind.ControlPoints[i,:]<np.array(HYGO_params.Control_range[i][0])) + np.sum(new_ind.ControlPoints[i,:]>np.array(HYGO_params.Control_range[i][1]))) == 0
                    valid = int(np.sum(np.array(valid))) == len(valid)

                    # Check if there is a custom validity function
                    if hasattr(HYGO_params,'validity') and callable(HYGO_params.validity):
                        custom_valid = HYGO_params.validity(new_ind.parameters)
                        valid = valid and custom_valid

                    # Update the information of the individuals in the population
                    if not exists:
                        for rep in range(HYGO_params.repetitions):
                            self.idx_to_evaluate[rep] = self.idx_to_evaluate[rep] + [nind]
                            
                        # Update information in the data structure
                        self.data.loc[nind,'Individuals'] = idx
                        self.data.loc[nind,('Parents','first')]  = int(id)
                        self.data.loc[nind,('Parents','second')] = int(idx1)
                        self.data.loc[nind,('Operation','Type')] = 'Shrinked'
                        self.data.loc[nind,('Operation','Point_parts')]  = 'None'
                        
                        if valid:
                            # Evaluate the individual and update the table
                            HYGO_table,checker = self.evaluate_population(copy.deepcopy(self.idx_to_evaluate),HYGO_params,HYGO_table,path,True)
                        else:
                            if HYGO_params.verbose:
                                print('Individual not evaluated, it was labeled as not valid')
                            self.data.loc[nind,'Costs'] = HYGO_params.badvalue
                            HYGO_table.individuals[idx].cost = HYGO_params.badvalue
                            if (HYGO_params.check_type=='Neval' or HYGO_params.limit_evaluations) and len(HYGO_table.individuals)>=HYGO_params.neval:
                                checker = False
                            else:
                                checker=True
                    else:
                        checker = True
                        
                    if not checker:
                        break
                    
        # Update the number of individuals in the population
        if checker:
            self.Nind+=1
        else:
            self.Nind = int(self.data.shape[0])

        return HYGO_table,checker

    def simplex_expanse(self,HYGO_params,HYGO_table,path,Reflection_idx):
        """
        Applies the expanse operation to the simplex.

        The expanse operation involves expanding the simplex away from the worst individual, creating a new individual
        with the expanded parameters, and adding it to the population. The operation is performed based on the parameters
        of the reflected individual.

        Args:
            HYGO_params (object): An object containing parameters for the genetic algorithm.
            HYGO_table (object): Table containing the population of individuals.
            path (str): The path to save additional information about the operations performed.
            Reflection_idx (int): Index of the reflected individual in the population.

        Returns:
            tuple: A tuple containing the updated HYGO_table and a boolean value (checker) indicating whether the maximum
            umber of evaluations was reached.
        """
        # Get the parameters of the reflected individual
        ind_exp = HYGO_table.individuals[Reflection_idx]

        # Display information if verbose mode is enabled
        if HYGO_params.verbose:
                print('    -Expanse:')
            
        # Compute centroid
        if (HYGO_params.optimization == 'Parametric') or (HYGO_params.optimization == 'Control Law' and not HYGO_params.SimplexInterpolation):
            centroid = np.array(self.simplex_centroid)

        if HYGO_params.optimization == 'Parametric':
            to_expand = np.array(ind_exp.parameters)
            # Compute expansion
            expanded = centroid + HYGO_params.expansion_gamma*(to_expand-centroid)

            # Ensure new parameters are inside bounds   
            expanded = self.check_params(expanded.tolist(),HYGO_params)

            # Create a new individual with the expanded parameters
            new_ind = Individual()
            new_ind.create(HYGO_params=HYGO_params,params=expanded)

            # Add the new individual to the table
            [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_ind)

            # Assume that the individual is valid
            valid = True

            # Check if there is a custom validity function
            if hasattr(HYGO_params,'validity') and callable(HYGO_params.validity):
                custom_valid = HYGO_params.validity(new_ind.parameters)
                valid = valid and custom_valid

            # Get the index of the individual in the population
            nind = self.data.shape[0]

            # Update the information of the individual in the population
            if not exists:
                for rep in range(HYGO_params.repetitions):
                        self.idx_to_evaluate[rep] = self.idx_to_evaluate[rep] + [nind]
                
                # Update information in the data structure
                self.data.loc[nind,'Individuals'] = idx
                self.data.loc[nind,('Parents','first')]  = int(self.simplex_idx[-1])
                self.data.loc[nind,('Parents','second')] = -1
                self.data.loc[nind,('Operation','Type')] = 'Expanded'
                self.data.loc[nind,('Operation','Point_parts')]  = 'None'

                if valid:
                    # Evaluate the individual and update the HYGO_table
                    HYGO_table,checker = self.evaluate_population(copy.deepcopy(self.idx_to_evaluate),HYGO_params,HYGO_table,path,True)
                else:
                    if HYGO_params.verbose:
                        print('Individual not evaluated, it was labeled as not valid')
                    self.data.loc[nind,'Costs'] = HYGO_params.badvalue
                    HYGO_table.individuals[idx].cost = HYGO_params.badvalue
                    if (HYGO_params.check_type=='Neval' or HYGO_params.limit_evaluations) and len(HYGO_table.individuals)>=HYGO_params.neval:
                        checker = False
                    else:
                        checker = True
            else:
                checker = True
                
        elif HYGO_params.optimization == 'Control Law':
            if HYGO_params.SimplexInterpolation: #Perform interpolation
                #Obtain the law
                law,new_parents,new_coeff = self.interpolate_simplex(HYGO_params,HYGO_table,'Expanded')
                
                # Create the new individual
                new_ind = Individual()
                new_ind.parameters = law
                new_ind.evaluate_ControlPoints(HYGO_params)
                
                checker = True # Flag to check if a valid mutation is found
                counter = 1 # Counter for limiting the number of attempts to find a valid mutation
                while checker:
                    # re-generate the matrix
                    new_ind = regenerate_matrix(HYGO_params,new_ind.ControlPoints)
                    # Store the parents and soefficients
                    new_ind.simplex_parents = new_parents
                    new_ind.coefficients = new_coeff
                    # Add the new individual to the table
                    [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_ind)
                    
                    checker = HYGO_params.remove_duplicates and counter<HYGO_params.MaxTries and exists
                    counter +=1
            else:
                to_expand = np.array(ind_exp.ControlPoints)
                # Compute expansion
                expanded = 3*np.array(centroid) - 2*to_expand
                # Create the new individual
                new_ind = Individual()
                
                checker = True # Flag to check if a valid mutation is found
                counter = 1 # Counter for limiting the number of attempts to find a valid mutation
                while checker:
                    # re-generate the matrix
                    new_ind = regenerate_matrix(HYGO_params,expanded)
                    # Add the new individual to the table
                    [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_ind)
                    
                    checker = HYGO_params.remove_duplicates and counter<HYGO_params.MaxTries and exists
                    counter +=1
                
            # Get the index of the individual in the population
            nind = self.data.shape[0]
            
            # Check if the control is within bounds
            valid = [0]*HYGO_params.control_outputs
            for i in range(HYGO_params.control_outputs):
                valid[i] = int(np.sum(new_ind.ControlPoints[i,:]<np.array(HYGO_params.Control_range[i][0])) + np.sum(new_ind.ControlPoints[i,:]>np.array(HYGO_params.Control_range[i][1]))) == 0
            valid = int(np.sum(np.array(valid))) == len(valid)

            # Check if there is a custom validity function
            if hasattr(HYGO_params,'validity') and callable(HYGO_params.validity):
                custom_valid = HYGO_params.validity(new_ind.parameters)
                valid = valid and custom_valid

            # Update the information of the individual in the population
            if not exists:
                for rep in range(HYGO_params.repetitions):
                        self.idx_to_evaluate[rep] = self.idx_to_evaluate[rep] + [nind]
                
                # Update information in the data structure
                self.data.loc[nind,'Individuals'] = idx
                self.data.loc[nind,('Parents','first')]  = int(self.simplex_idx[-1])
                self.data.loc[nind,('Parents','second')] = -1
                self.data.loc[nind,('Operation','Type')] = 'Expanded'
                self.data.loc[nind,('Operation','Point_parts')]  = 'None'
                
                if valid:
                    # Evaluate the individual and update the table
                    HYGO_table,checker = self.evaluate_population(copy.deepcopy(self.idx_to_evaluate),HYGO_params,HYGO_table,path,True)
                else:
                    if HYGO_params.verbose:
                        print('Individual not evaluated, it was labeled as not valid')
                    self.data.loc[nind,'Costs'] = HYGO_params.badvalue
                    HYGO_table.individuals[idx].cost = HYGO_params.badvalue
                    if (HYGO_params.check_type=='Neval' or HYGO_params.limit_evaluations) and len(HYGO_table.individuals)>=HYGO_params.neval:
                        checker = False
                    else:
                        checker=True
            else:
                checker = True

        # Update the number of individuals in the population
        if checker:
             self.Nind+=1

        return HYGO_table,checker

    def simplex_batch_cyle(self,HYGO_params,HYGO_table,path):
        """
        Computes a simplex cycle evaluation with a batch procedure which consists on building all
        possible individuals with all simplex operations and evaluate them in batches. Once evaluated, 
        perform the simplex procedure to determine which information is saved

        Args:
            HYGO_params (object): An object containing parameters for the genetic algorithm.
            HYGO_table (object): Table containing the population of individuals.
            path (str): The path to save additional information about the operations performed.

        Returns:
            tuple: A tuple containing the updated HYGO_table and a boolean value (checker) indicating whether the maximum number 
            of individuals was reached.
        """

        # Get costs of the worst and second-to-last individuals in the simplex
        J1 = self.simplex_costs[0]
        J_end_minus_1 = self.simplex_costs[-2]

        # Identify best individual
        ind1 = HYGO_table.individuals[int(self.simplex_idx[0])]
        # Identify the worst individual
        ind_end = HYGO_table.individuals[int(self.simplex_idx[-1])]

        # Compute centroid
        if (HYGO_params.optimization == 'Parametric') or (HYGO_params.optimization == 'Control Law' and not HYGO_params.SimplexInterpolation):
            centroid = np.array(self.simplex_centroid)

        if HYGO_params.optimization == 'Parametric':
            # Create the index to evaluate list
            idx_to_evaluate = []
            # Compute the reflected parameters
            reflected = centroid + HYGO_params.reflection_alpha*(centroid-np.array(ind_end.parameters))
            # Ensure the new parameters are inside the bounds
            reflected = self.check_params(reflected.tolist(),HYGO_params)
            # Create a new individual with the reflected parameters
            HYGO_table, ind_refl, idx_refl, exists_refl, valid_refl = self.create_individual(HYGO_table,HYGO_params,reflected,'Reflection',[int(self.simplex_idx[-1]),-1])
            idx_to_evaluate.append(self.data.shape[0]-1)
            # Compute expansion
            expanded = centroid + HYGO_params.expansion_gamma*(np.array(reflected)-centroid)
            # Ensure new parameters are inside bounds   
            expanded = self.check_params(expanded.tolist(),HYGO_params)
            # Create a new individual with the expanded parameters
            HYGO_table, ind_exp, idx_exp, exists_exp, valid_exp = self.create_individual(HYGO_table,HYGO_params,expanded,'Expanded',[int(self.simplex_idx[-1]),-1])
            idx_to_evaluate.append(self.data.shape[0]-1)
            # Compute the contracted parameters
            contracted1 = centroid + HYGO_params.contraction_rho*(np.array(reflected)-centroid) 
            contracted2 = centroid + HYGO_params.contraction_rho*(np.array(ind_end.parameters)-centroid) 
            # Ensure the new parameters are inside the bounds
            contracted1 = self.check_params(contracted1.tolist(),HYGO_params)
            contracted2 = self.check_params(contracted2.tolist(),HYGO_params)

            # Create a new individual with the contracted parameters
            HYGO_table, ind_contr1, idx_contr1, exists_contr1, valid_contr1 = self.create_individual(HYGO_table,HYGO_params,contracted1,'Contraction',[int(self.simplex_idx[-1]),-1])
            idx_to_evaluate.append(self.data.shape[0]-1)
            HYGO_table, ind_contr2, idx_contr2, exists_contr2, valid_contr2 = self.create_individual(HYGO_table,HYGO_params,contracted2,'Contraction',[int(self.simplex_idx[-1]),-1])
            idx_to_evaluate.append(self.data.shape[0]-1)

            # Compute shrink
            params1 = ind1.parameters
            # Get the indices of individuals to shrink
            ids = self.simplex_idx[1:]
            shrinked_inds = []
            shrinked_idx = []
            shrinked_exists = []
            shrinked_valid = []

            # Perform the shrink operation for each individual in the simplex (except the best individual)
            for id in ids:
                params = HYGO_table.individuals[int(id)].parameters
                shrink = np.array(params1) + HYGO_params.shrinkage_sigma*(np.array(params)-np.array(params1))
                shrink = self.check_params(shrink.tolist(),HYGO_params)
                HYGO_table, new_indiv, idx, exists, valid = self.create_individual(HYGO_table,HYGO_params,shrink,'Shrinked',[self.simplex_idx[0],id])
                idx_to_evaluate.append(self.data.shape[0]-1)

                shrinked_inds.append(new_indiv)
                # Add the new individual to the table
                shrinked_idx.append(idx)
                shrinked_exists.append(exists)
                shrinked_valid.append(valid)

            inds = np.array([ind_refl,ind_exp,ind_contr1,ind_contr2] + shrinked_inds)
            idxs = np.array([idx_refl,idx_exp,idx_contr1,idx_contr2] + shrinked_idx)
            exists = np.array([exists_refl,exists_exp,exists_contr1,exists_contr2] + shrinked_exists)
            valid = np.array([valid_refl,valid_exp,valid_contr1,valid_contr2] + shrinked_valid)

            # Only evaluate individuals that did not exist previously and individuals that are valid
            evaluate = np.logical_and(np.array(np.abs(exists-1),dtype='bool'),valid)

            idx_to_evaluate = np.array(idx_to_evaluate)[evaluate]
            # Eliminate redundancies in case an individual was already present in the table
            idx_to_evaluate = np.unique(idx_to_evaluate).tolist()

        elif HYGO_params.optimization == 'Control Law':
            # Create the index to evaluate list
            idx_to_evaluate = []
            if HYGO_params.SimplexInterpolation:
                # Create the new individual
                HYGO_table, ind_refl, idx_refl, exists_refl, valid_refl = self.create_individual(HYGO_table,HYGO_params,None,'Reflection',[int(self.simplex_idx[-1]),-1])
                idx_to_evaluate.append(self.data.shape[0]-1)

                # Create the new individual
                HYGO_table, ind_exp, idx_exp, exists_exp, valid_exp = self.create_individual(HYGO_table,HYGO_params,None,'Expanded',[int(self.simplex_idx[-1]),-1])
                idx_to_evaluate.append(self.data.shape[0]-1)

                # Create a new individual with the contracted parameters
                HYGO_table, ind_contr1, idx_contr1, exists_contr1, valid_contr1 = self.create_individual(HYGO_table,HYGO_params,None,'Contraction',[int(self.simplex_idx[-1]),-1])
                idx_to_evaluate.append(self.data.shape[0]-1)

                # Compute shrink
                params1 = ind1.ControlPoints
                # Get the indices of individuals to shrink
                ids = self.simplex_idx[1:]
                shrinked_inds = []
                shrinked_idx = []
                shrinked_exists = []
                shrinked_valid = []

                # Perform the shrink operation for each individual in the simplex (except the best individual)
                for i in range(HYGO_params.SimplexSize-1):
                    # Obtain the individual
                    HYGO_table, new_indiv, idx, exists, valid = self.create_individual(HYGO_table,HYGO_params,None,'Shrinked',i+1,[self.simplex_idx[0],ids[i]])
                    idx_to_evaluate.append(self.data.shape[0]-1)

                    shrinked_inds.append(new_indiv)
                    # Add the new individual to the table
                    shrinked_idx.append(idx)
                    shrinked_exists.append(exists)
                    shrinked_valid.append(valid)
                
                inds = np.array([ind_refl,ind_exp,ind_contr1] + shrinked_inds)
                idxs = np.array([idx_refl,idx_exp,idx_contr1] + shrinked_idx)
                exists = np.array([exists_refl,exists_exp,exists_contr1] + shrinked_exists)
                valid = np.array([valid_refl,valid_exp,valid_contr1] + shrinked_valid)
            else:
                # Get parameters of the worst individual and the centroid of the simplex
                to_reflect = np.array(ind_end.ControlPoints)
                # Compute the reflected parameters
                reflected = 2*np.array(centroid) - to_reflect
                # Create the new individual
                HYGO_table, ind_refl, idx_refl, exists_refl, valid_refl = self.create_individual(HYGO_table,HYGO_params,reflected,'Reflection',[int(self.simplex_idx[-1]),-1])
                idx_to_evaluate.append(self.data.shape[0]-1)

                # Compute expansion
                to_expand = np.array(ind_exp.ControlPoints)
                # Compute expansion
                expanded = 3*np.array(centroid) - 2*to_expand
                # Create the new individual
                HYGO_table, ind_exp, idx_exp, exists_exp, valid_exp = self.create_individual(HYGO_table,HYGO_params,expanded,'Expanded',[int(self.simplex_idx[-1]),-1])
                idx_to_evaluate.append(self.data.shape[0]-1)

                # Compute the contracted parameters
                ind_contr1 = np.array(ind_refl.ControlPoints)
                ind_contr2 = np.array(ind_end.ControlPoints)
                
                # Compute the contracted parameters
                contracted1 = 0.5*(np.array(centroid) + ind_contr1)
                contracted2 = 0.5*(np.array(centroid) + ind_contr2)

                # Create a new individual with the contracted parameters
                HYGO_table, ind_contr1, idx_contr1, exists_contr1, valid_contr1 = self.create_individual(HYGO_table,HYGO_params,contracted1,'Contraction',[int(self.simplex_idx[-1]),-1])
                idx_to_evaluate.append(self.data.shape[0]-1)
                HYGO_table, ind_contr2, idx_contr2, exists_contr2, valid_contr2 = self.create_individual(HYGO_table,HYGO_params,contracted2,'Contraction',[int(self.simplex_idx[-1]),-1])
                idx_to_evaluate.append(self.data.shape[0]-1)

                # Compute shrink
                shrinked_inds = []
                shrinked_idx = []
                shrinked_exists = []
                shrinked_valid = []

                ids = self.simplex_idx[1:]
                # Perform the shrink operation for each individual in the simplex (except the best individual)
                for i in range(HYGO_params.SimplexSize-1):
                    # Obtain the individual to shrink Control points
                    params = HYGO_table.individuals[int(id)].ControlPoints
                    # Compute the shrink operation
                    shrink = 0.5*(np.array(params)+np.array(params1))
                    HYGO_table, new_indiv, idx, exists, valid = self.create_individual(HYGO_table,HYGO_params,shrink,'Shrinked',[self.simplex_idx[0],ids[i]])
                    idx_to_evaluate.append(self.data.shape[0]-1)

                    shrinked_inds.append(new_indiv)
                    # Add the new individual to the table
                    shrinked_idx.append(idx)
                    shrinked_exists.append(exists)
                    shrinked_valid.append(valid)

                #inds = np.array([ind_refl,ind_exp,ind_contr1,ind_contr2] + shrinked_inds)
                idxs = np.array([idx_refl,idx_exp,idx_contr1,idx_contr2] + shrinked_idx)
                inds = []
                for idx in idxs:
                    inds.append(HYGO_table.individuals[int(idx)])
                inds = np.asarray(inds)
                exists = [exists_refl,exists_exp,exists_contr1,exists_contr2] + shrinked_exists
                valid = np.array([valid_refl,valid_exp,valid_contr1,valid_contr2] + shrinked_valid)
            
            # Only evaluate individuals that did not exist previously and individuals that are valid
            evaluate = np.logical_and(np.array(np.abs(exists-1),dtype='bool'),valid)

            idx_to_evaluate = np.array(idx_to_evaluate)[evaluate]
            # Eliminate redundancies in case an individual was already present in the table
            idx_to_evaluate = np.unique(idx_to_evaluate).tolist()

        # Update list of indexes to evaluate
        for rep in range(HYGO_params.repetitions):
            self.idx_to_evaluate[rep] = self.idx_to_evaluate[rep] + idx_to_evaluate

        # Update index and sort pop index
        self.data.reset_index(drop=True,inplace=True)
        self.sort_pop()
        # Evaluate the individual and update the HYGO_table
        HYGO_table,checker = self.evaluate_population(copy.deepcopy(self.idx_to_evaluate),HYGO_params,HYGO_table,path,True)

        # Obtain the costs
        costs = []
        for idx in idxs:
            costs.append(HYGO_table.individuals[int(idx)].cost)

        # List of individuals to eliminate
        inds_to_eliminate = []
        idx_to_eliminate = []
        # Perform the simplex operation
        if checker:
            if HYGO_params.verbose:
                print('    -Reflection')
            if np.array(costs[0]) >= np.array(J_end_minus_1):
                if HYGO_params.verbose:
                    print('    -Single Contraction')
                # Eliminate the expansion ind
                inds_to_eliminate.append(inds[1])
                idx_to_eliminate.append(idxs[1])
                if (HYGO_params.optimization == 'Parametric') or (HYGO_params.optimization == 'Control Law' and not HYGO_params.SimplexInterpolation):
                    # Contraction
                    if costs[0]<self.simplex_costs[-1]:
                        # Eliminate the second contracted
                        inds_to_eliminate.append(inds[3])
                        idx_to_eliminate.append(idxs[3])
                        # Check if shrink required
                        if not costs[2]>=self.simplex_costs[-1]:
                            inds_to_eliminate.extend(inds[4:])
                            idx_to_eliminate.extend(idxs[4:])
                        else:
                            if HYGO_params.verbose:
                                print('    -Shrink')
                    else:
                        # Eliminate the third contracted
                        inds_to_eliminate.append(inds[2])
                        idx_to_eliminate.append(idxs[2])
                        # Check if shrink required
                        if not costs[3]>=self.simplex_costs[-1]:
                            inds_to_eliminate.extend(inds[4:])
                            idx_to_eliminate.extend(idxs[4:])
                        else:
                            if HYGO_params.verbose:
                                print('    -Shrink')
                elif HYGO_params.optimization == 'Control Law' and  HYGO_params.SimplexInterpolation:
                    # Check if shrink
                    if not costs[2]>=self.simplex_costs[-1]:
                            inds_to_eliminate.extend(inds[3:])
                            idx_to_eliminate.extend(idxs[3:])
                    else:
                        if HYGO_params.verbose:
                            print('    -Shrink')

            elif np.array(costs[0]) < np.array(J1):
                if HYGO_params.verbose:
                    print('    -Expanse')
                # Reflection-Expansion
                inds_to_eliminate = inds[2:]
                idx_to_eliminate = idxs[2:]
            else:
                # Only Reflection
                inds_to_eliminate = inds[1:]
                idx_to_eliminate = idxs[1:]
        

        idx_to_eliminate = np.flip(np.unique(idx_to_eliminate))

        eliminate_from_table = []
        for idx in idx_to_eliminate:
            pos = np.where(idxs == idx)[0].tolist()
            for p in pos:
                eliminate_from_table.append(exists[pos].tolist()[0])

        for i,idx in enumerate(idx_to_eliminate):
            self.data.drop(index=self.data[self.data['Individuals']==idx].index,labels=None, axis=0, inplace=True,columns=None, level=None, errors='raise')
            # If it did not exist previous to the simplex, eliminate it from the table
            if not eliminate_from_table[i]:
                #TODO: CHANGED
                if HYGO_params.individual_paths:
                    paths_to_change = HYGO_table.individuals[int(idx_to_eliminate[i])].path
                    for pp in paths_to_change:
                        tries = 0
                        renamed = False
                        while not renamed:
                            if not os.path.isdir(pp+f'_discarded_{tries}'):
                                os.rename(pp, pp+f'_discarded_{tries}')
                                renamed = True
                            else:
                                tries +=1
                HYGO_table.remove_individual(int(idx_to_eliminate[i]))
                

        # Update index and sort pop index
        self.data.reset_index(drop=True,inplace=True)
        self.sort_pop()

        # Update the indexes of the individuals that were not eliminated
        for i,ind in enumerate(inds):
            if ind not in inds_to_eliminate:
                new_idx = HYGO_table.find_indiv(ind)
                row_idx = self.data[self.data['Individuals']==idxs[i]].index
                self.data.iloc[row_idx,0] = copy.deepcopy(new_idx)
                #TODO: CHANGED
                if HYGO_params.individual_paths:
                    paths_to_change = ind.path
                    for pp in paths_to_change:
                        temp = pp.split('/')[-1]
                        temp = temp[10:]
                        len_to_eliminate = len(str(temp))
                        ppp = pp[:-len_to_eliminate] + str(new_idx)
                        if pp==ppp:
                            continue
                        else:
                            os.rename(pp, ppp)
                            HYGO_table.individuals[new_idx].path.append(ppp)

        # Sort the population
        self.sort_pop()
        self.Nind = self.data.shape[0]
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

    def check_convergence(self,HYGO_params,HYGO_table):
        """
        Checks for convergence of the genetic algorithm based on a specified convergence criterion.

        Args:
            HYGO_params (object): An object containing parameters for the genetic algorithm.
            HYGO_table (object): Table containing the population of individuals.

        Returns:
            bool: True if convergence criterion is met, otherwise False.
        """

        if HYGO_params.check_type=='interval':
            interval = []
            
            # Compute the interval threshold for each prameter
            for i in range(HYGO_params.N_params):
                if type(HYGO_params.Nb_bits)!=int:
                    if len(HYGO_params.Nb_bits)>1:
                        interval.append((HYGO_params.params_range[i][1]-HYGO_params.params_range[i][0])/(2**HYGO_params.Nb_bits[i]-1))
                    else:
                        interval.append((HYGO_params.params_range[i][1]-HYGO_params.params_range[i][0])/(2**HYGO_params.Nb_bits[0]-1))
                else:
                    interval.append((HYGO_params.params_range[i][1]-HYGO_params.params_range[i][0])/(2**HYGO_params.Nb_bits-1))
        else:
            return False

        # Iterate through individuals in the population to check for convergence
        for ind in HYGO_table.individuals:
            params = ind.parameters
            
            if HYGO_params.check_type=='interval':
                convergence = []
                global_minima = np.array(HYGO_params.global_minima)
                
                # Check if the parameters are within the specified interval of the global minima
                if len(global_minima.shape)>1:
                    for minima in global_minima:
                        convergence = []
                        for i in range(len(params)):
                            if (params[i]<(minima[i]+HYGO_params.ninterval*interval[i])) and (params[i]>(minima[i]-HYGO_params.ninterval*interval[i])):
                                convergence.append(True)
                            else:
                                convergence.append(False)
                        if (len(convergence)-sum(convergence))==0:
                            print('---->Early convergence')
                            return True
                else:
                    for i in range(len(params)):
                        if (params[i]<(HYGO_params.global_minima[i]+HYGO_params.ninterval*interval[i])) and (params[i]>(HYGO_params.global_minima[i]-HYGO_params.ninterval*interval[i])):
                            convergence.append(True)
                        else:
                            convergence.append(False)
                    if (len(convergence)-sum(convergence))==0:
                        print('---->Early convergence')
                        return True
        
        return False

    def fixation_checker(self,simplex_values,HYGO_params,HYGO_table,path,previous_checker):
        '''
        Checks if, for the last n simplex sizes, the simplex individuals were stuck in
        a hyperplane of dimension n-1. The fixation checker assesses the possibility of
        a fixation in the population by examining the residuals of a hyperplane fit to
        the current simplex individuals' parameters. If the residuals exceed a specified
        threshold, the function correctsthe fixation by introducing a new individual based
        on the hyperplane correction by creating it in the normal direction from the centroid.
        If the simplex has not changed in the last cycles it indicates that the simplex may
        have gotten stuck and itintroduces a new individual with random parameters.
        The checking process helps tomaintain diversity in the population and prevents
        fixation in specific regions of the search space.

        Parameters:
            - simplex_values (list): A list containing the indices of individuals in the last n simplex.
            - HYGO_params (object): An object containing parameters for the genetic algorithm.
            - HYGO_table (object): Table containing the information of the individuals.
            - path (str): The path to the directory where the GA algorithm is executed.
            - previous_checker (bool): The result of the previous fixation check, indicating
                                    whether the population is currently in a fixation state.

        Returns:
            - HYGO_table (object): Updated table containing the population of individuals.
            - checker (bool): The result of the fixation check after the function is executed.
            - idx_inserted (bool): Indicates whether a new individual has been inserted into the
                                population during the fixation check.
        '''
        coefficients = []
        residual = []
        checker = previous_checker
        force_random=False
        idx_inserted=False

        equal_checker = []
        
        idx_simplex = []

        # Extracted indices from simplex values
        for simplex in simplex_values:
            # Check if the simplex has been repeated in the last cycles
            equal_checker.append(np.sum(np.array(simplex_values[0])-np.array(simplex)))
            
            # Save all of the individual's indexes in the last n simplex
            for idx in simplex:
                idx_simplex.append(idx)

        # Eliminate redundancies
        idx_simplex = np.unique(idx_simplex)

        # Extract parameters from individuals
        parameters = []
        for idx in idx_simplex:
            parameters.append(np.array(HYGO_table.individuals[int(idx)].parameters))

        parameters = np.transpose(np.array(parameters))

        try:
            # Fit a hyperplane to the parameters
            coefficients,_ = curve_fit(self.func, parameters[0:-1], parameters[-1],
                                np.random.rand(len(parameters)))

            # Compute the residual
            residual = self.R2(parameters,coefficients)
        except:
            residual=0
    
        # If the data fits in a hyperplane and the last n cycles have not been equal, perform the hyperplane correction
        if residual>HYGO_params.Simplex_R2 and not np.sum(equal_checker)==0:
            # Apply hyperplane correction
            coefficients[-1] = -1
            coefficients = coefficients/np.linalg.norm(coefficients)

            # Calculate movement based on coefficients
            interval = []
            for i in range(HYGO_params.N_params):
                if type(HYGO_params.Nb_bits)!=int:
                    if len(HYGO_params.Nb_bits)>1:
                        interval.append((HYGO_params.params_range[i][1]-HYGO_params.params_range[i][0])/(2**HYGO_params.Nb_bits[i]-1))
                    else:
                        interval.append((HYGO_params.params_range[i][1]-HYGO_params.params_range[i][0])/(2**HYGO_params.Nb_bits[0]-1))
                else:
                    interval.append((HYGO_params.params_range[i][1]-HYGO_params.params_range[i][0])/(2**HYGO_params.Nb_bits-1))

            # Generate the new parameters by taking the normal direction to the hyperplane and moving a 
            #   specified number of intervals from the centroid
            new_params = []
            for i,inter in enumerate(interval):
                new_params.append(float(self.simplex_centroid[i]+coefficients[i]*HYGO_params.Simplex_intervals_movement*inter))

            # Adjust the params so they are valid and not outside the domain
            new_params = self.check_params(new_params,HYGO_params)

            # Create the new individual and add it to the population
            new_ind = Individual()
            new_ind.create(HYGO_params=HYGO_params,params=new_params)

            [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_ind)

            # Update the number of individuals in the population
            nind = self.data.shape[0]
            # Add to population data
            if not exists:
                for rep in range(HYGO_params.repetitions):
                    self.idx_to_evaluate[rep] = self.idx_to_evaluate[rep] + [nind]
                
                # Update data in population
                self.data.loc[nind,'Individuals'] = idx
                self.data.loc[nind,('Parents','first')]  = int(self.simplex_idx[-1])
                self.data.loc[nind,('Parents','second')] = -1
                self.data.loc[nind,('Operation','Type')] = 'Correction'
                self.data.loc[nind,('Operation','Point_parts')]  = 'None'
                
                if HYGO_params.verbose:
                    print('    -Hyperplane Correction')
                
                # Evaluate the individual
                HYGO_table,checker = self.evaluate_population(copy.deepcopy(self.idx_to_evaluate),HYGO_params,HYGO_table,path,True)
            
                # Update simplex if the individual is not new
                self.simplex_idx[-1] = idx
                self.simplex_costs[-1] = HYGO_table.individuals[idx].cost

                # Order the simplex, compute the centroid and add it to the memory of the simplex
                self.simplex_ordering()
                self.simplex_centroid_computation(HYGO_params,HYGO_table)

                # Update simplex memory with the new simplex for the current cycle
                self.simplex_memory[-1] = self.simplex_idx

                # Indicate that an individual was created through hyperplane correction
                idx_inserted = True
            else:
                # If the individual existed create an individual by random movement
                force_random=True

        # Create an individual if the last n simplexes have been equal or if the hyperplane correction
        #   created an already existing individual
        elif np.sum(equal_checker)==0 or force_random:
            # Apply random correction based on a number of intervals movement
            interval = []
            for i in range(HYGO_params.N_params):
                if type(HYGO_params.Nb_bits)!=int:
                    if len(HYGO_params.Nb_bits)>1:
                        interval.append((HYGO_params.params_range[i][1]-HYGO_params.params_range[i][0])/(2**HYGO_params.Nb_bits[i]-1))
                    else:
                        interval.append((HYGO_params.params_range[i][1]-HYGO_params.params_range[i][0])/(2**HYGO_params.Nb_bits[0]-1))
                else:
                    interval.append((HYGO_params.params_range[i][1]-HYGO_params.params_range[i][0])/(2**HYGO_params.Nb_bits-1))
            
            exists = True
            ntries = 1

            # Retry introducing a random individual until it is successful or reaches the maximum tries
            while exists and ntries<HYGO_params.MaxTries:
                new_params = []
                ninter = HYGO_params.Simplex_intervals_random_movement
                for i,inter in enumerate(interval):
                    new_params.append(float(self.simplex_centroid[i]+np.random.randint(low=-ninter,high=ninter,size=1)*inter))

                new_params = self.check_params(new_params,HYGO_params)

                new_ind = Individual()
                new_ind.create(HYGO_params=HYGO_params,params=new_params)

                [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_ind)

                ntries+=1
            
            # Update the number of individuals in the population
            nind = self.data.shape[0]
            # Add to population data
            if not exists:
                for rep in range(HYGO_params.repetitions):
                    self.idx_to_evaluate[rep] = self.idx_to_evaluate[rep] + [nind]
                
                # Update population data
                self.data.loc[nind,'Individuals'] = idx
                self.data.loc[nind,('Parents','first')]  = int(self.simplex_idx[-1])
                self.data.loc[nind,('Parents','second')] = -1
                self.data.loc[nind,('Operation','Type')] = 'CorrectionRandom'
                self.data.loc[nind,('Operation','Point_parts')]  = 'None'
                
                if HYGO_params.verbose:
                    print('    -Individual repetition correction')
                    
                # Evaluate the individual and update the table
                HYGO_table,checker = self.evaluate_population(copy.deepcopy(self.idx_to_evaluate),HYGO_params,HYGO_table,path,True)
            
            # Update simplex if the individual is not new
            self.simplex_idx[-1] = idx
            self.simplex_costs[-1] = HYGO_table.individuals[idx].cost

            # Order the simplex, compute the centroid and add it to the memory of the simplex
            self.simplex_ordering()
            self.simplex_centroid_computation(HYGO_params,HYGO_table)

            # Update simplex memory with the new simplex for the current cycle
            self.simplex_memory[-1] = self.simplex_idx

            # Indicate that an individual was created through hyperplane correction
            idx_inserted = True
            
        # Make a security backup if specified
        if HYGO_params.security_backup:
            import dill
            file = open(path+'/Gen'+str(self.generation)+'/pop_backup.obj','wb')
            dill.dump(self,file)
            file.close()
            file = open(path+'/Gen'+str(self.generation)+'/table_backup.obj','wb')
            dill.dump(HYGO_table,file)
            file.close()

        return HYGO_table,checker,idx_inserted

    def create_individual(self,HYGO_table,HYGO_params,params,operation,parents,shrink_idx=None):
        '''
        Creastes and individual, checks its params and adds it to the table and the population data

        Args:
            HYGO_params (object): An object containing parameters for the genetic algorithm.
            HYGO_table (object): Table containing the population of individuals.
            params (list): parameters of the individual to be created
            operation (str): operation by which was created
            parents (list): list with the indexes of the parents used to create the individual
            shrink_idx(int): idx of the shrink individual, only in interpolation control law

        Returns:
            HYGO_table
            ind (onject): created individual
            idx (int): index of the individual
            exists (bool): indicates if the individual already existed
            valid (bool): indicates if an individual is within control threshold. Only used in
                control law optimisation
        '''

        if HYGO_params.optimization == 'Parametric':
            # Create the individual
            ind = Individual()
            ind.create(HYGO_params=HYGO_params,params=params)
            # Add the new individual to the table
            [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=ind)
            # Get the index of the individual in the population
            nind = self.data.shape[0]
            # Assume individual is valid
            valid = True

            # Check if there is a custom validity function
            if hasattr(HYGO_params,'validity') and callable(HYGO_params.validity):
                custom_valid = HYGO_params.validity(ind.parameters)
                valid = valid and custom_valid

            if not exists:
                # Update information in the data structure
                self.data.loc[nind,'Individuals'] = idx
                self.data.loc[nind,('Parents','first')]  = int(parents[0])
                self.data.loc[nind,('Parents','second')] = int(parents[1])
                self.data.loc[nind,('Operation','Type')] = operation
                self.data.loc[nind,('Operation','Point_parts')]  = 'None'

                if not valid:
                    if HYGO_params.verbose:
                        print('Individual not evaluated, it was labeled as not valid')
                    self.data.loc[nind,'Costs'] = HYGO_params.badvalue
                    HYGO_table.individuals[idx].cost = HYGO_params.badvalue
                
        elif HYGO_params.optimization == 'Control Law':
            if HYGO_params.SimplexInterpolation:
                if operation!='Shrinked':
                    law,new_parents,new_coeff = self.interpolate_simplex(HYGO_params,HYGO_table,operation)
                else:
                    law,new_parents,new_coeff = self.interpolate_simplex(HYGO_params,HYGO_table,operation,shrink_idx)
                # Create the new individual
                ind = Individual()
                ind.parameters = law
                ind.evaluate_ControlPoints(HYGO_params)
                
                checker = True # Flag to check if a valid mutation is found
                counter = 1 # Counter for limiting the number of attempts to find a valid mutation
                while checker:
                    # re-generate the matrix
                    ind = regenerate_matrix(HYGO_params,ind.ControlPoints)
                    # Store the parents and soefficients
                    ind.simplex_parents = new_parents
                    ind.coefficients = new_coeff
                    # Add the new individual to the table
                    [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=ind)
                    
                    checker = HYGO_params.remove_duplicates and counter<HYGO_params.MaxTries and exists
                    counter +=1
            else:
                # Create the new individual
                ind = Individual()
                
                checker = True # Flag to check if a valid mutation is found
                counter = 1 # Counter for limiting the number of attempts to find a valid mutation
                while checker:
                    # re-generate the matrix
                    ind = regenerate_matrix(HYGO_params,params)
                    # Add the new individual to the table
                    [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=ind)
                    
                    checker = HYGO_params.remove_duplicates and counter<HYGO_params.MaxTries and exists
                    counter +=1

            # Get the index of the individual in the population
            nind = self.data.shape[0]
            
            # Check if the control is within bounds
            valid = [0]*HYGO_params.control_outputs
            for i in range(HYGO_params.control_outputs):
                valid[i] = int(np.sum(ind.ControlPoints[i,:]<np.array(HYGO_params.Control_range[i][0])) + np.sum(ind.ControlPoints[i,:]>np.array(HYGO_params.Control_range[i][1]))) == 0
            valid = int(np.sum(np.array(valid))) == len(valid)

            # Check if there is a custom validity function
            if hasattr(HYGO_params,'validity') and callable(HYGO_params.validity):
                custom_valid = HYGO_params.validity(ind.parameters)
                valid = valid and custom_valid

            # Add the individual to the population and table data if it does not exist
            if not exists:
                # Update information in the data structure
                self.data.loc[nind,'Individuals'] = idx
                self.data.loc[nind,('Parents','first')]  = int(self.simplex_idx[-1])
                self.data.loc[nind,('Parents','second')] = -1
                self.data.loc[nind,('Operation','Type')] = operation
                self.data.loc[nind,('Operation','Point_parts')]  = 'None'

                if not valid:
                    if HYGO_params.verbose:
                        print('Individual not evaluated, it was labeled as not valid')
                    self.data.loc[nind,'Costs'] = HYGO_params.badvalue
                    HYGO_table.individuals[idx].cost = HYGO_params.badvalue

        return HYGO_table, ind, idx, exists, valid

    @staticmethod
    def func(X, *argv):
        '''
        Computes the value of a hyperplane in n dimensions, where n is the length of X.

        Parameters:
            - X (list or array): The input vector representing the coordinates in the hyperplane.
            - *argv (float): Coefficients of the hyperplane.

        Returns:
            - float: The computed value of the hyperplane at the given coordinates.
        '''
        # Sum the product of each coordinate and its corresponding coefficient
        add = 0
        for i in range(len(X)):
            add += argv[i]*X[i]

        return add + argv[-1]

    def R2(self,parameters,coeffs):
        '''
        Calculates the R-squared value, a measure of the goodness of fit, for a hyperplane fitting.

        Parameters:
            - parameters (array): Array containing the parameters of the hyperplane (coordinates and output values).
            - coeffs (array): Coefficients of the hyperplane.

        Returns:
            - float: The R-squared value indicating the goodness of fit for the hyperplane.
        '''
        # Calculate residuals (the differences between actual and predicted output)
        residuals = parameters[-1]- self.func(parameters[0:-1], *coeffs)
        
        # Sum of squared residuals
        ss_res = np.sum(residuals**2)

        # Total sum of squares
        ss_tot = np.sum((parameters[-1]-np.mean(parameters[-1]))**2)

        # Calculate R-squared value nd return it
        return 1 - ss_res/ss_tot
    
    def interpolate_simplex(self,HYGO_params,HYGO_table,operation,shrink_idx=None):
        '''
        Method that performs the linear interpolation of individuals in the simplex
        to compute the new operation. It is based in ref [1]
        
        Args:
            HYGO_params (object): An object containing parameters for the genetic algorithm.
            HYGO_table (object): Table containing the population of individuals.
            path (str): The path to save additional information about the operations performed.
            shrink_idx(int): idx of the shrink individual
            
        Returns:
            interpolated_law(str): interpolated law.
            new_simplex_parents(list): list of the simplex parents of the individuals taht have coeffcients != 0
            new_coeff(list): list of the coefficients that built the law
        
        References: 
            [1] Cornejo Maceda, G., Lusseyran, F., & Noack, B. (2022). xMLC - A Toolkit for Machine Learning Control (vol. vol. 2). doi:10.24355/DBBS.084-202208220937-0
        '''
        
        # Obtrain simplexes parents
        simplex_parents = []
        unique_parents = []
        coefficients = []
        for idx in self.simplex_idx:
            ind = HYGO_table.individuals[int(idx)]
            simplex_parents.append(ind.simplex_parents)
            unique_parents += ind.simplex_parents
            coefficients.append(ind.coefficients)
        
        # Obtain the unique individuals
        unique_parents = np.sort(np.unique(unique_parents))
        
        # Initialize the matrix
        Matrix = np.zeros((unique_parents.size,HYGO_params.SimplexSize))
        
        # Fill the matrix with the coefficients
        for i in range(HYGO_params.SimplexSize):
            parents = simplex_parents[i]
            coeff = coefficients[i]
            for j in range(len(parents)):
                position = np.where(unique_parents==parents[j])[0][0]
                Matrix[position,i] = coeff[j]
                
        # Obtain the control laws
        control_laws = np.empty((HYGO_params.control_outputs,unique_parents.size),dtype=object)
        for i in range(unique_parents.size):
            ind = HYGO_table.individuals[int(unique_parents[i])]
            control_laws[:,i] = np.array(ind.parameters)
            
        # Compute the centroid vector, the last individual is excluded
        Centroid = np.ones((HYGO_params.SimplexSize,1))/(HYGO_params.SimplexSize-1)
        Centroid[-1] = 0
        
        # Compute the coefficient matrix
        Matrix = Matrix.T
        
        # Obtain the coefficients of the first individual
        coeff1 = Matrix[0,:]
        
        # Set the vector for the last individual
        coeffend = np.zeros((HYGO_params.SimplexSize,1))
        coeffend[-1] = 1
        # ALGO MAL AQUI
        # Compute the coefficients
        if operation=='Reflection':
            new_coeff = np.matmul(np.transpose(2*Centroid-coeffend),Matrix)
        elif operation=='Expanded':
            new_coeff = np.matmul(np.transpose(3*Centroid-2*coeffend),Matrix)
        elif operation=='Contraction':
            new_coeff = np.matmul(np.transpose(Centroid+coeffend),Matrix/2)
        elif operation=='Shrinked':
            new_coeff = (coeff1 + Matrix[shrink_idx,:])/2
        
        # Eliminate second dimension
        if len(new_coeff.shape)>1:
            new_coeff = new_coeff[0]
        
        # Initialize the strings
        new_law = ['']*HYGO_params.control_outputs
        
        # computation of the control law
        for i in range(HYGO_params.control_outputs):
            # Variable for controling the number of laws inserted
            inserted = 0
            for j in range(len(new_coeff)):
                if new_coeff[j]==0:
                    continue
                elif inserted==0:
                    new_law[i]+=str(new_coeff[j])+'*'+control_laws[i,j]
                    inserted+=1
                else:
                    new_law[i]+='+('+str(new_coeff[j])+'*'+control_laws[i,j]+')'
                    inserted+=1
        
        # Remove parents and coefficients that do not contribute
        new_simplex_parents = unique_parents[new_coeff!=0]
        new_coeff = new_coeff[new_coeff!=0]
        
                    
        return new_law, new_simplex_parents.tolist(), new_coeff.tolist()