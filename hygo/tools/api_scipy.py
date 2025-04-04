import numpy as np
from ..individual import *

class API_Scipy:
    """
    The API-Scipy class represents the integration of some optimization methods
    from scipy within the exploitation phase of HYGO.

    Attributes:
        cma_es_state (str): Current state of the simplex optimization process.

    Methods:
        __init__: Initializes the Simplex object.
        
    """

    def __init__(self):
        """
        Initializes the Simplex object.
        """
        self.cma_es_state = None
        pass

    def Scipy_exploitation(self,HYGO_params,HYGO_table,path):
        """
        Executes the exploitation phase using the selected optimization method.

        Args:
            HYGO_params (object): An object containing genetic algorithm parameters.
            HYGO_table (object): The  table containing information about individuals.
            path (str): The path where the results are stored.

        Returns:
            Tuple[PopulationTable, bool, bool]: A tuple containing the updated population table, a flag indicating whether the maximum number of individuals is reached, and a flag indicating convergence.
        """
        if HYGO_params.verbose:
            print(f'---> Scipy {HYGO_params.Scipy_method} method selected for generation '+str(self.generation))
            
        if HYGO_params.Scipy_method=='Minimize':
            HYGO_table,checker,convergence = self.Scipy_minimize(HYGO_params,HYGO_table,path)
        
        self.scipy_state = 'Scipy Done'
        return HYGO_table,checker,convergence
            
    def Scipy_minimize(self,HYGO_params,HYGO_table,path):
        """
        Executes the exploitation with the minimize function.

        Args:
            HYGO_params (object): An object containing genetic algorithm parameters.
            HYGO_table (object): The  table containing information about individuals.
            path (str): The path where the results are stored.

        Returns:
            Tuple[PopulationTable, bool, bool]: A tuple containing the updated population table, a flag indicating whether the maximum number of individuals is reached, and a flag indicating convergence.
        """
        from scipy.optimize import minimize
        
        # Select the Initial condition
        if HYGO_params.Scipy_Initial_Condition == 'Best':
            idx = self.data.loc[0,'Individuals']
        elif HYGO_params.Scipy_Initial_Condition == 'Random':
            idx = np.random.choice(self.data.shape[0],1)
            idx = self.data.loc[idx,'Individuals']
        
        config = copy.deepcopy(HYGO_params.Scipy_options)
        if 'x0' in config:
            config.pop('x0')
            print('Warning: Initial condition from Scipy options ignored')
        
        options = {}
        if 'disp' not in config:
            options['disp'] = HYGO_params.verbose
        else:
            options['disp'] = config['disp']
            config.pop('disp')
        
        if 'maxiter' not in config:
            config['maxiter'] = self.data.shape[0]
            print('Warning: Number of evaluations not specified so the number of evaluations was limited to the same as the current population size')
        else:
            options['maxiter'] = config['maxiter']
            config.pop('maxiter')
        
        config['bounds'] = (np.array(HYGO_params.params_range))
        
        # Get the initial condition
        x0 = HYGO_table.individuals[int(idx)].parameters
        
        if hasattr(HYGO_params,'Scipy_force_evaluate') and HYGO_params.Scipy_force_evaluate:
            force=True
        else:
            force=False
        
        # Define the Cost function
        def Cost_function(x, HYGO_table=HYGO_table):
            # Get the individual index
            n_ind = int(self.data.shape[0])
            
            new_ind = Individual()
            new_ind.create(HYGO_params=HYGO_params,params=x.tolist())
            
            # Add individual to the table
            [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_ind)
            
            # Update value if exists
            if exists and not force:
                return HYGO_table.individuals[idx].cost
            if force:
                HYGO_table.individuals[idx].parameters = x.tolist()
            
            # Assume that the individual is valid
            valid = True

            # Check if there is a custom validity function
            if hasattr(HYGO_params,'validity') and callable(HYGO_params.validity):
                valid = HYGO_params.validity(new_ind.parameters)
            
            if not exists:
                self.data.loc[n_ind,'Individuals'] = idx
                self.data.loc[n_ind,('Parents','first')] = -1
                self.data.loc[n_ind,('Parents','second')] = -1
                self.data.loc[n_ind,('Operation','Type')] = 'Scipy'
                self.data.loc[n_ind,('Operation','Point_parts')] = str('None')
                
                if not valid:
                    self.data.loc[n_ind,['Costs']] = HYGO_params.badvalue
                    HYGO_table.individuals[idx].cost = HYGO_params.badvalue
                    
                    return HYGO_params.badvalue
                
            # Update indexes that have to be evaluated
                for rep in range(HYGO_params.repetitions):
                    self.idx_to_evaluate[rep] = self.idx_to_evaluate[rep] + [n_ind]
            else:
                for rep in range(HYGO_params.repetitions):
                    location = int(np.where(self.data['Individuals'].values.astype(int)==int(idx))[0][0])
                    self.data.loc[location,'Costs'] = -1
                    self.data.loc[location,('Operation','Type')] = 'Scipy'
                    self.idx_to_evaluate[rep] = self.idx_to_evaluate[rep] + [location]
                
            # Evaluate the individual
            HYGO_table,checker = self.evaluate_population(copy.deepcopy(self.idx_to_evaluate),HYGO_params,HYGO_table,path,True)
            
            self.Nind = int(self.data.shape[0])
            
            return HYGO_table.individuals[idx].cost
        
        # Call the optimization
        result = minimize(fun=Cost_function, x0=x0, options=options, **config)
        
        # Check if the max number evaluation is reached
        checker = True
        if (hasattr(HYGO_params,'check_type') and HYGO_params.check_type=='Neval') or HYGO_params.limit_evaluations:
            if len(HYGO_table.individuals>=HYGO_params.neval):
                checker = False
                
        # Check convergence
        if HYGO_params.check_convergence:
            convergence = self.check_convergence(HYGO_params,HYGO_table)
            
        
        return [HYGO_table,checker,convergence]

        