__author__ = 'Isaac Robledo Martín'

import numpy as np
import copy

from .tools.chromosome_to_params import chromosome_to_params
# Define the custom operation functions

def mydiv(arg0,arg1):
    if (type(arg0)==float or type(arg0)==int) and (type(arg1)==float or type(arg1)==int):
        return arg0/arg1 if arg1!=0 else 1e36
    else:
        return np.where(np.array(arg1) != 0, np.divide(arg0,arg1), 1e36)

def mylog(arg0):
    if type(arg0)==float or type(arg0)==int:
        return float(np.log10(arg0)) if arg0>0 else 1e36
    else:
        return np.where(np.array(arg0) > 0, np.log10(arg0), 1e36)

def myexp(arg0):
    if type(arg0)==float or type(arg0)==int:
        return float(np.exp(arg0)) if arg0<83 else 1e36
    else:
        return np.where(np.array(arg0) <83, np.log10(arg0), 1e36)

def mytanh(arg0):
    if type(arg0)==float or type(arg0)==int:
        return float(np.tanh(arg0))
    else:
        return np.tanh(arg0)

def mycos(arg0):
    if type(arg0)==float or type(arg0)==int:
        return float(np.cos(arg0))
    else:
        return np.cos(arg0)

def mysin(arg0):
    if type(arg0)==float or type(arg0)==int:
        return float(np.sin(arg0))
    else:
        return np.sin(arg0)

def mynanchecker(arg):
    try:
        arg = float(arg)
        return np.isnan(arg)
    except:
        # This corresponds to the case where a sensor has been introduced
        return False

def isnumeric(arg):
    try:
        float(arg)
        return True
    except:
        # This corresponds to the case where a sensor has been introduced
        return False

def myfloat(arg):
    try:
        return float(arg)
    except:
        # This corresponds to the case where a sensor has been introduced
        return arg

class Individual():
    '''
    A class representing an individual in the genetic algorithm.

    Attributes:
        - chromosome (list): The genetic material of the individual in binary, representing a solution.
        - cost (float): The cost or fitness score of the individual, indicating its performance.
        - parameters (list): Genetic material of the individual converted into float point values.
        - path (list): Individual path of the individual where information can be saved or accesed in the fitness function.
        - occurrences (int): The number of occurrences of this individual in the optimisation.
        - hash (str): A hash value uniquely identifying the individual.
        - ControlPoints (np.ndarray): Values of the control function at the specified eval time samples and the sensors. 
                Only present in the Control Law optimization
        - simplex_parents (list): list of simplex indexes used to create the individual. Only present in
                the Control Law optimization with control law interpolation
        - coefficients (list): list of weights used to create the individual if the interpolation is on. Only present in
                the Control Law optimization with control law interpolation
        - reconstruction_time (float): time taken to reconstruct an individual. Only present
                in the Control Law optimization.

    Methods:
        - create(HYGO_params,params): Initializes the chromosome randomly or using a specific set of parameters based on the Plant parameters in HYGO_params.
        - create_chromosome(HYGO_params): Creates a random chromosome based on the Plan parameters HYGO_params.
        - eliminate_introns(HYGO_params,chromosome): eliminates the introns for an introduced chromosome for a control law.
        - chromosome_to_control_law(HYGO_params,chromosome): Translates the introduced chromosome into a control law in a specific format.
        - simplify_my_law(HYGO_params,old_law): Simplifies the introduced law by precomputing values using introduced simplifications.
        - law_to_function(HYGO_params,law): Translates the control law into an evaluable string by Python.
        - chromosome_to_function(HYGO_params,chromosome): Combines other methods to convert the control law chromosome into an evaluable string.
        - evaluate_ControlPoints(HYGO_params): computes the values of the control laws at the specified time and sensor values
        - mutate(HYGO_params,ind): [Static Method] Applies a mutation strategy to the introduced individual.
        - crossover(HYGO_params,ind1,ind2): [Static Method] Performs crossover between two individuals to create offspring.

    Example Usage:
        # Create an individual 
        individual1 = Individual()

        # Initialize chromosome randomly
        individual1.create(HYGO_params)

        # Create an individual 
        individual2 = Individual()

        # Initialize the chromosome with a set of parameters
        individual2.create(HYGO_params,params)

        # Perform the crossover operator between two individuals
        new1,new2,operation = Individual.crossover(HYGO_params,individual1,individual2)

        # Create a new individual through mutation
        new3,instructions = Individuals.mutate(HYGO_params,individual1)
    '''

    def __init__(self):
        '''
        Initialize the Individual.
        '''

        self.chromosome       = []
        self.cost             = None
        self.parameters       = []
        self.path             = []
        self.ocurrences       = 1
        self.hash             = None

    def create(self, HYGO_params, params=None):
        '''
    Method for creating an individual.
    
    In the parametric optimization case if specific params are introduced
    the chromosome will be created with them, if not, it will be created
    randomly.

    Parameters:
        - HYGO_params (object): An object containing parameters for the Genetic Algorithm.
        - params (list, optional): Specific parameters for creating the chromosome. Defaults to None.
    '''
        # Create a random chromosome using the specified HYGO_params
        chromosome = self.create_chromosome(HYGO_params)

        if HYGO_params.optimization == 'Parametric':
            if params:
                # If specific params are provided, convert them to a chromosome
                from .tools.params_to_chromosome import params_to_chromosome

                # If custom parameters are specified, call the specific function round_individuals
                if hasattr(HYGO_params,'custom_parameters'):
                    from .tools.round_params import round_params
                    params = round_params(HYGO_params,params)

                # Translate the introduced parameters to chromosome
                chromosome = params_to_chromosome(HYGO_params,params) 
            else:
                # If no specific params are provided, convert the chromosome to params
                params = chromosome_to_params(HYGO_params,chromosome)

                # If custom parameters are specified, call the specific function round_individuals
                if hasattr(HYGO_params,'custom_parameters'):
                    from .tools.round_params import round_params
                    from .tools.params_to_chromosome import params_to_chromosome
                    params = round_params(HYGO_params,params)

                    # Translate the introduced parameters to chromosome
                    chromosome = params_to_chromosome(HYGO_params,params)
                    
            # Set the individual's parameters, chromosome, and hash value
            self.parameters = params
            self.chromosome = chromosome
            self.hash = hash(tuple(self.chromosome))
            
        elif HYGO_params.optimization == 'Control Law':            
            # Eliminate the introns
            chromosome = self.eliminate_introns(HYGO_params,chromosome)
            
            # Regenerate if the chroosome is empty and specified
            if HYGO_params.remove_duplicates:
                counter = 0
                checker = chromosome.shape[0]<HYGO_params.Minimum_instructions
                
                while checker:
                    # Create a random chromosome using the specified HYGO_params
                    chromosome = self.create_chromosome(HYGO_params)
                    
                    # Eliminate the introns
                    chromosome = self.eliminate_introns(HYGO_params,chromosome)

                    # Check if success
                    counter += 1
                    checker = chromosome.shape[0]<HYGO_params.Minimum_instructions
            
            # Obtain the control law
            law = self.chromosome_to_control_law(HYGO_params,chromosome)
            # Simplify the control law
            simplified_law = [self.simplify_my_law(HYGO_params,law[i]) for i in range(len(law))]
            # Convert the law into interpretable
            params = [self.law_to_function(HYGO_params,simplified_law[i]) for i in range(len(simplified_law))]
            # Set the individual's parameters, chromosome, and hash value
            self.parameters = params
            self.chromosome = chromosome
            self.hash = hash(tuple(self.chromosome.flatten().tolist()))
            # Obtain the control points
            self.evaluate_ControlPoints(HYGO_params)
            if HYGO_params.exploitation:
                self.reconstruction_time = np.nan    

    def create_chromosome(self, HYGO_params):
        '''
        Create a random chromosome based on the introduced Plant parameters.

        Parameters:
            - HYGO_params (object): An object containing parameters for the Genetic Algorithm.

        For the parametric optimization:
        
        Returns:
            - chromosome(numpy.ndarray): A randomly generated binary chromosome.
            
        For the control law optimization creates a chromosome of size HYGO_params.Ninstructions_initialx4 wich
        represents the control law.
        
        Returns:
            - chromosome(numpy.ndarray): matrix representing the control law.
        
        '''
        
        if HYGO_params.optimization == 'Parametric':
            # Extract Nb_bits and N_params from HYGO_params
            Nb_bits = HYGO_params.Nb_bits
            N_params = HYGO_params.N_params

            # Check if Nb_bits is an integer or a list
            if type(Nb_bits) != int:
                if len(Nb_bits) > 1:
                    # If Nb_bits is a list with more than one element, generate a binary chromosome with the sum of Nb_bits
                    chromosome = np.random.randint(2, size=np.sum(Nb_bits))
                else:
                    # If Nb_bits is a list with a single element, generate a binary chromosome with Nb_bits[0] * N_params size
                    chromosome = np.random.randint(2, size=Nb_bits[0] * N_params)
            else:
                # If Nb_bits is an integer, generate a binary chromosome with Nb_bits * N_params size
                chromosome = np.random.randint(2, size=Nb_bits * N_params)
        elif HYGO_params.optimization == 'Control Law':
            # Obtain the number of initial instructions
            Ninst = int(np.random.rand(1)[0]*(HYGO_params.Ninstructions_initial_max-HYGO_params.Ninstructions_initial_min)) + HYGO_params.Ninstructions_initial_min

            # Create the chromosome randomly
            chromosome = np.zeros((Ninst,4))
            chromosome[:,0] = np.random.randint(0,HYGO_params.register_size-1,(Ninst)).astype(int)
            chromosome[:,1] = np.random.randint(0,HYGO_params.register_size-1,(Ninst)).astype(int)
            chromosome[:,2] = np.random.randint(0,HYGO_params.number_operations,(Ninst)).astype(int)
            chromosome[:,3] = np.random.randint(0,HYGO_params.variable_registers-1,(Ninst)).astype(int)

        return chromosome

    def eliminate_introns(self,HYGO_params,chromosome):
        '''
        This function is in carge of eliminating the operations (rows of the chromosome)
        that are not useful for the control law. The algorithm 3.1 from [1] is implemented
        
        Parameters:
            - HYGO_params(object):  An object containing genetic algorithm parameters.
            - chromosome(np.array): Ninstx4 matrix containing the control law represented by integers [1] 
        
        Returns:
            - new_chromosome(np.array): updated cromosome without introns
            
        References:
            [1] M. Brameier and W. Banzhaf. Linear Genetic Programming. Springer Science & Business Media, 2006.
        '''
        
        operations = HYGO_params.operations
        
        # Create the new chromosome
        new_chromosome = copy.deepcopy(chromosome)
        
        output_registers = list(range(HYGO_params.control_outputs))
        
        # Initialize the effective registers as the output ones
        Reff = copy.deepcopy(output_registers)
        
        for i in range(chromosome.shape[0]-1,-1,-1):
            if np.sum(chromosome[i,3]==np.array(Reff))>0:
                comparison = np.array(np.array(Reff)==chromosome[i,3])
                # Obtain the indexes and order them in descending order so we can use .pop()
                [Reff.pop(idx) for idx in np.flip(np.sort(np.array(range(comparison.size))[comparison]))]
                
                Reff += [int(chromosome[i,0])]
                if operations['n_args'][int(chromosome[i,2])]==2:
                    Reff += [int(chromosome[i,1])]
            else:
                new_chromosome = np.delete(new_chromosome,i,0)
                
        return new_chromosome

    def chromosome_to_control_law(self,HYGO_params,chromosome):
        '''
        Function that converts the matrix of Ninstx4 (the chromosome) into a 
        interpretable control law following the format explained in [1].
        
        Parameters:
            - HYGO_params(object):  An object containing genetic algorithm parameters.
            - chromosome(np.array): Ninstx4 matrix containing the control law represented by integers [1] 
            
        Returns:
            -law(str): interpetable control law.
        
        This function is inspired by:
            [1] M. Brameier and W. Banzhaf. Linear Genetic Programming. Springer Science & Business Media, 2006.
        '''
        
        registers = copy.deepcopy(HYGO_params.registers)
        operations = HYGO_params.operations
        
        for i in range(chromosome.shape[0]):
            arg0 = int(chromosome[i,0])
            arg1 = int(chromosome[i,1])
            op = int(chromosome[i,2])
            output = int(chromosome[i,3])
            
            if operations['n_args'][op] == 1: # Only one argument functions
                registers[output] = '(' + operations['op'][op] + ' ' + registers[arg0] + ')'
            else: # Two argument functions
                registers[output] = '(' + operations['op'][op] + ' ' + registers[arg0] + ' ' + registers[arg1] + ')'
                        
        return registers[0:HYGO_params.control_outputs]

    def simplify_my_law(self,HYGO_params,old_law):
        '''
        This function simplifies the obtained control law obtained by paring the chromosome
        of an individual. It works by identifying each node and its arguments and the
        simplifications impremented in the HYGO_params.operations are applied.
        
        Parameters:
            - HYGO_params(object):  An object containing genetic algorithm parameters.
            - old_law(str): String of the control law in the format specified in [1]
            
        Returns:
            -new_law(str): simplified control law.
        
        This function is inspired by:
            [1] M. Brameier and W. Banzhaf. Linear Genetic Programming. Springer Science & Business Media, 2006.
            [2] Cornejo Maceda, G., Lusseyran, F., & Noack, B. (2022). xMLC - A Toolkit for Machine Learning Control (vol. vol. 2). doi:10.24355/DBBS.084-202208220937-0
        '''
        
        operations = HYGO_params.operations
        precission = HYGO_params.precission
        
        # Find the indeces where the spaces are located
        comparison = np.array([val==' ' for val in old_law])
        indexes = np.array(range(comparison.size))
                        
        spaces = indexes[comparison]
        
        if spaces.size==0:
            return old_law # Simplification finalised
        else:
            operator = old_law[1:spaces[0]] # Obtain the first operator

            '''
            operator_spaces is a list locating the indexes of the highest level parenthesis.
            This code was obtained from [2]
            Ex:             (+ (- a b) b)
            cumsum of (     1112222222222
            cumsum of )     0000000001112
            difference      1112222221110
            space           0010010100100
            space*diff      0010020200100
            space*diff==1   0010000000100
            
            The last index is placed so the arguments are always between teo indexes
            '''
            operators_spaces = np.array(range(len(old_law)))[np.multiply(np.cumsum(np.array([val=='(' for val in old_law]))-np.cumsum(np.array([val==')' for val in old_law])),
                                comparison.astype(int))==1].tolist() + [len(old_law)-1]
                    
            new_law = old_law
            
            for i in range(len(operations['op'])):
                if operations['op'][i]==operator:
                    for j in range(operations['n_args'][i]):
                        exec('arg'+str(j)+'='+'self.simplify_my_law(HYGO_params,old_law[operators_spaces[j]+1:operators_spaces[j+1]])')
                        new_law = eval('new_law.replace(old_law[operators_spaces[j]+1:operators_spaces[j+1]],'+'arg'+str(j)+')')
                    for j in range(len(operations['simplification_cond'][i])):
                        if eval(operations['simplification_cond'][i][j]):
                            new_law = eval(operations['simplification_action'][i][j])
                            if type(new_law)==float or 'nan' in new_law:
                                print(operations['simplification_action'][i][j])
            
            return new_law 

    def law_to_function(self,HYGO_params,law):
        '''
        Converts each control law into a string that is evaluable by python
        
        Parameters:
            Parameters:
            - HYGO_params(object):  An object containing genetic algorithm parameters.
            - law(str): String of the control law in the format specified in [1].
            
        Returns:
            -new_law(str): string evaluable by python.
            
        References:
            [1] M. Brameier and W. Banzhaf. Linear Genetic Programming. Springer Science & Business Media, 2006.
        '''
        
        operations = HYGO_params.operations
        
        # Find the indeces where the spaces are located
        comparison = np.array([val==' ' for val in law])
        indexes = np.array(range(comparison.size))
                        
        spaces = indexes[comparison]
        
        if spaces.size == 0:
            return law
        else:
            # Same procedure as in simplify_my_law function
            operator = law[1:spaces[0]]
            operators_spaces = np.array(range(len(law)))[np.multiply(np.cumsum(np.array([val=='(' for val in law]))-np.cumsum(np.array([val==')' for val in law])),
                                comparison.astype(int))==1].tolist() + [len(law)-1]
            
            new_law = 0
            
            for i in range(len(operations['op'])):
                if operations['op'][i]==operator:
                    new_law = operations['expression'][i]
                    for j in range(operations['n_args'][i]):
                        result = self.law_to_function(HYGO_params,law[operators_spaces[j]+1:operators_spaces[j+1]])
                        new_law = new_law.replace('arg'+str(j),result)
            
            return new_law

    def chromosome_to_function(self,HYGO_params,chromosome):
        '''
        Function utilized to translate a control law chromosome into an
        avaluable python string.
        
        Parameters:
            - HYGO_params(object):  An object containing genetic algorithm parameters.
            - chromosome(np.array): Ninstx4 matrix containing the control law represented by integers
        
        Returns:
            - law(str): string evaluable by python.
        '''
        
        # Eliminate the introns
        chromosome = self.eliminate_introns(HYGO_params,chromosome)
        # Obtain the control law
        law = self.chromosome_to_control_law(HYGO_params,chromosome)
        # Simplify the control law
        simplified_law = [self.simplify_my_law(HYGO_params,law[i]) for i in range(len(law))]
        # Convert the law into interpretable
        law = [self.law_to_function(HYGO_params,simplified_law[i]) for i in range(len(simplified_law))]
        
        return law

    def evaluate_ControlPoints(self,HYGO_params):
        '''
        Obtains the value of the control law at the specified moments of time and at the
        selected sensor values and stores the in the attribute IndividualControlPoints
        
        Parameters:
            - HYGO_params(object):  An object containing genetic algorithm parameters.
        '''
        # Initialize the control points
        self.ControlPoints = np.zeros((HYGO_params.control_outputs,HYGO_params.N_control_points))
        # Iterate through each control output
        for i in range(len(self.parameters)):
            try:
                law = self.parameters[i]
                b = lambda t,s : eval(law)
                '''for j in range(HYGO_params.N_control_points):
                    # Obtain the sensor values
                    sensor_Values = [HYGO_params.ControlPoints[k,j] for k in range(HYGO_params.control_inputs)]
                    # Evaluate the control law
                    self.ControlPoints[i,j] = b(HYGO_params.evaluation_time_sample[j],sensor_Values)'''
                self.ControlPoints[i,:] = b(HYGO_params.evaluation_time_sample,HYGO_params.ControlPoints)
            except:
                self.ControlPoints[i,:] = np.ones(HYGO_params.N_control_points)*HYGO_params.badvalue
    
    @staticmethod
    def crossover(HYGO_params,ind1,ind2):

        '''
        Perform crossover operation between two individuals.

        Parameters:
            - HYGO_params (object): An object containing parameters for the Genetic Algorithm.
            - ind1 (Individual): The first individual participating in crossover.
            - ind2 (Individual): The second individual participating in crossover.

        Returns:
            - tuple: Two new individuals resulting from crossover, and information about the crossover operation.

        '''
        if HYGO_params.optimization == 'Parametric':
            # Extract chromosomes from the individuals
            chrom1 = ind1.chromosome
            chrom2 = ind2.chromosome

            # Combine the chromosomes into a list for easier manipulation
            chrom = [chrom1, chrom2]

            # Get the length of the chromosomes
            l_chromosomes = len(chrom1)

            # Get the number of crossover points
            N_divisions = HYGO_params.crossover_points

            # Check if the number of divisions exceeds the chromosome length - 1
            if N_divisions > (l_chromosomes - 1):
                raise ValueError('The crossover number of divisions exceeds the limit.')

            # Determine crossover points based on the crossover_mix option
            # It is important to note that with N_divisions number of divisions,
            #   there are N_divisions+1 number of segments
            if HYGO_params.crossover_mix:
                '''
                If the crossover mix option is selected an intercalated list of indexes
                    which indicate to which individual each segment corresponds. It should
                    be noted that the first segment corresponds to the other individual always
                    and is forced in a later stage.
                    For example, with one division the lists created would be:
                        rchromosome1 = [0] The second segment of the first ind corresponds
                                            to the first individual
                        rchromosome2 = [1]
                    If the number of divisions was three, the lists would be:
                        rchromosome1 = [0,1,0]
                        rchromosome2 = [1,0,1]
                '''
                rchromosome1 = np.mod(1+np.linspace(1,N_divisions,N_divisions),2)
                rchromosome2 = np.mod(np.linspace(1,N_divisions,N_divisions),2)
            else:
                '''
                Creates lists of the indexes corresponding to each individual randomly.
                For example, with one division the lists created may be:
                        rchromosome1 = [1]
                        rchromosome2 = [1] Since it is random, both may coincide
                    If the number of divisions was three, the lists may be:
                        rchromosome1 = [0,1,1]
                        rchromosome2 = [1,1,0]
                '''
                rchromosome1 = np.random.randint(2,size=N_divisions)-1
                rchromosome2 = np.random.randint(2,size=N_divisions)-1

            # As commented, force that the first segment of the new individuals
            #   corrspond to the opposite index (new1 has the first segment from ind2)
            rchromosome1 = np.insert(rchromosome1,0,1)
            rchromosome2 = np.insert(rchromosome2,0,0)

            # Randomly select the indexes where each of the segments will end
            crossover_pts = np.unique(np.random.randint(l_chromosomes,size=N_divisions)).tolist()
            # Introduce the end of the chromosome as the end of the last segment
            crossover_pts.append(l_chromosomes+1)

            # Initialize indices for selecting crossover masks
            chrom1_idx = []
            chrom2_idx = []

            # Apply crossover masks to select chromosome segments
            idx = 0
            for i in range(l_chromosomes):
                if i<crossover_pts[idx]:
                    chrom1_idx.append(rchromosome1[idx])
                    chrom2_idx.append(rchromosome2[idx])
                else:
                    idx+=1
                    chrom1_idx.append(rchromosome1[idx])
                    chrom2_idx.append(rchromosome2[idx])

            # Construct new chromosomes by combining selected segments
            chromosome1 = [chrom[int(chrom1_idx[i])][i] for i in range(l_chromosomes)]
            chromosome2 = [chrom[int(chrom2_idx[i])][i] for i in range(l_chromosomes)]

            # Check if the resulting chromosomes have the correct length
            if len(chromosome1)!=l_chromosomes or len(chromosome2)!=l_chromosomes:
                raise ValueError('Crossover error')

            # Convert chromosomes to parameters
            params1 = chromosome_to_params(HYGO_params,chromosome1)
            params2 = chromosome_to_params(HYGO_params,chromosome2)

            # Round and convert parameters back to chromosomes if custom parameters are defined
            if hasattr(HYGO_params,'custom_parameters'):
                from .tools.round_params import round_params
                from .tools.params_to_chromosome import params_to_chromosome

                params1 = round_params(HYGO_params,params1)
                chromosome1 = params_to_chromosome(HYGO_params,params1)

                params2 = round_params(HYGO_params,params2)
                chromosome2 = params_to_chromosome(HYGO_params,params2)

            # Create new Individual objects for the offspring
            new1 = Individual()
            new2 = Individual()

            # Assign data to the new individuals
            new1.chromosome = chromosome1
            new1.parameters = params1
            new1.hash = hash(tuple(new1.chromosome))
            new2.chromosome = chromosome2
            new2.parameters = params2
            new2.hash = hash(tuple(new2.chromosome))

            # Collect information about the crossover operation
            operation = [crossover_pts,chrom]
            
        elif HYGO_params.optimization == 'Control Law':
            
            # Extract chromosomes from the individuals
            chromosome1 = ind1.chromosome
            chromosome2 = ind2.chromosome
            
            # Obtain the number of instructions of each indiv
            N1 = chromosome1.shape[0]
            N2 = chromosome2.shape[0]      
            
            # Obtain the smaller size
            N0 = np.min([N1,N2])      
            
            # Get the number of crossover points
            N_divisions = HYGO_params.crossover_points
            
            # Get the maximum number of instructions
            Maxinst = HYGO_params.Max_instructions
            
            # Initialize values to ensure it goes into thw while
            new_N1 = Maxinst*2
            new_N2 = 0
            
            # Regenerate until both chromosomes are of the appropriate size
            while new_N1>Maxinst or new_N2>Maxinst or new_N1 <= HYGO_params.crossover_points or new_N2 <= HYGO_params.crossover_points:
                
                # Store the points for each individual
                crossover_pts = [[],[]]
                
                # Iterate through the number of instructions
                for i in range(N_divisions):
                    #Checker to see if the first value was added
                    inside = False

                    if N1>1:
                        value1 = np.random.randint(1,N1)
                    else:
                        value1 = np.random.randint(1)
                    if value1 not in crossover_pts[0]:
                        crossover_pts[0].append(value1)
                    else:
                        inside = True
                        
                    if N2>1:
                        value2 = np.random.randint(1,N2)
                    else:
                        value2 = np.random.randint(1)

                    # Only add the crossover point if the first one was added
                    if value2 not in crossover_pts[1] and not inside:
                        crossover_pts[1].append(value2)
                
                # Introduce the size of the cromosome to always have to limits
                crossover_pts[0].append(N1)
                crossover_pts[1].append(N2)
                
                # Sort the crossover points
                crossover_pts = np.sort(np.array(crossover_pts))
                
                # Obtain the actual number of instructions
                actual_N = crossover_pts.shape[0]-1 # Subtract the size gained by adding the last idx
                
                # Previously explained
                if HYGO_params.crossover_mix:
                    rchromosome1 = np.mod(1+np.linspace(1,actual_N,actual_N),2)
                    rchromosome2 = np.mod(np.linspace(1,actual_N,actual_N),2)
                else:
                    rchromosome1 = np.random.randint(2,size=actual_N)-1
                    rchromosome2 = np.random.randint(2,size=actual_N)-1
                
                rchromosome1 = np.insert(rchromosome1,0,1)
                rchromosome2 = np.insert(rchromosome2,0,0)
                
                # Save the segments of chromosomes
                sub_chromosomes = [[],[]]
                sub_chromosomes[0].append(chromosome1[0:crossover_pts[0,0]])
                sub_chromosomes[1].append(chromosome2[0:crossover_pts[1,0]])
                
                for i in range(actual_N):
                    sub_chromosomes[0].append(chromosome1[crossover_pts[0,i]:crossover_pts[0,i+1]])
                    sub_chromosomes[1].append(chromosome1[crossover_pts[1,i]:crossover_pts[1,i+1]])
                
                # Assign the instructions by vertically stacking the chromosomes
                new_chromosome1 = sub_chromosomes[int(rchromosome1[0])][0]
                new_chromosome2 = sub_chromosomes[int(rchromosome2[0])][0]
                
                for i in range(1,actual_N+1):
                    new_chromosome1 = np.vstack((new_chromosome1,sub_chromosomes[int(rchromosome1[i])][i]))
                    new_chromosome2 = np.vstack((new_chromosome2,sub_chromosomes[int(rchromosome2[i])][i]))
                
                new_N1 = new_chromosome1.shape[0]
                new_N2 = new_chromosome2.shape[0]
                
            # Create new Individual objects for the offspring
            new1 = Individual()
            new2 = Individual()
            
            # Assign data to the new individuals
            new1.chromosome = new_chromosome1
            new1.parameters = new1.chromosome_to_function(HYGO_params,new1.chromosome)
            new1.hash = hash(tuple(new1.chromosome.flatten().tolist()))
            
            # Obtain the control points
            new1.evaluate_ControlPoints(HYGO_params)  
            
            new2.chromosome = new_chromosome2
            new2.parameters = new2.chromosome_to_function(HYGO_params,new2.chromosome)
            new2.hash = hash(tuple(new2.chromosome.flatten().tolist()))
            
            # Obtain the control points
            new2.evaluate_ControlPoints(HYGO_params)

            if HYGO_params.exploitation:
                new1.reconstruction_time = np.nan
                new2.reconstruction_time = np.nan    
            
            operation = [crossover_pts,sub_chromosomes]
            

        return new1,new2,operation

    @staticmethod
    def mutate(HYGO_params,ind):
        '''
        Perform mutation operation on an individual.

        Parameters:
            - HYGO_params (object): An object containing parameters for the Genetic Algorithm.
            - ind (Individual): The individual to mutate.

        Returns:
            - ind (Individual): The individual to mutate.
            - instructions (list): List including the mutation positions.

        '''
        # Determine the type of mutation based on HYGO_params
        mutation_type = HYGO_params.mutation_type

        # Define mutation rates based on the mutation type
        mutation_selection = {
                              'at_least_one': 1/len(ind.chromosome)
                              }
        
        if hasattr(HYGO_params,'mutation_rate'):
            mutation_selection['classic'] = HYGO_params.mutation_rate
        
        # Select the mutation rate for the current type
        pm = mutation_selection[mutation_type]

        # Create a new Individual object to store the mutated individual
        new_ind = Individual()

        # Copy the original chromosome for mutation
        m_chromosome = np.array(ind.chromosome)

        
            # Ensure at least one mutation occurs
        check = True
        while check:
            # Generate random probabilities for each position in the chromosome
            probs = np.random.rand(len(m_chromosome))

            # Determine positions for mutation based on the probability
            instructions = probs < pm

            # Count the number of mutations
            number_mut = instructions.sum()

            # Check if at least one mutation occurred
            check = number_mut == 0
        
        if HYGO_params.optimization == 'Parametric':
            # Apply mutations to the selected positions
            m_chromosome[instructions] = np.random.randint(2,size=number_mut)

            # Convert the mutated chromosome to a list
            m_chromosome=m_chromosome.tolist()

            # Convert the mutated chromosome to parameters
            params = chromosome_to_params(HYGO_params,m_chromosome)

            # Round and convert parameters back to chromosomes if custom parameters are defined
            if hasattr(HYGO_params,'custom_parameters'):
                from .tools.round_params import round_params
                from .tools.params_to_chromosome import params_to_chromosome

                params = round_params(HYGO_params,params)
                m_chromosome = params_to_chromosome(HYGO_params,params)

            # Assign data to the new individual
            new_ind.chromosome = m_chromosome
            new_ind.parameters = params
            new_ind.hash = hash(tuple(new_ind.chromosome))
            
        elif HYGO_params.optimization == 'Control Law':
            
            # Apply mutations to the selected positions
            m_chromosome[instructions,0] = np.random.randint(0,HYGO_params.register_size-1,size=number_mut).astype(int)
            m_chromosome[instructions,1] = np.random.randint(0,HYGO_params.register_size-1,size=number_mut).astype(int)
            m_chromosome[instructions,2] = np.random.randint(0,HYGO_params.number_operations-1,size=number_mut).astype(int)
            m_chromosome[instructions,3] = np.random.randint(0,HYGO_params.variable_registers-1,size=number_mut).astype(int)
            
            # Assign data to the new individuals
            new_ind.chromosome = m_chromosome
            new_ind.parameters = new_ind.chromosome_to_function(HYGO_params,new_ind.chromosome)
            new_ind.hash = hash(tuple(new_ind.chromosome.flatten().tolist())) 
            
            # Obtain the control points
            new_ind.evaluate_ControlPoints(HYGO_params) 

            if HYGO_params.exploitation:
                new_ind.reconstruction_time = np.nan   

        # Return the mutated individual and a list indicating mutated positions
        return new_ind,instructions.tolist()

if __name__=='__main__':

    from ..examples.parameters_control_law import Parameters
    from .tools.chromosome_to_params import chromosome_to_params

    HYGO_params = Parameters

    ind1 = Individual()
    ind1.create(HYGO_params=HYGO_params,params=[1,1])
    ind2 = Individual()
    ind2.create(HYGO_params=HYGO_params,params=[8,32])

    ind3,ind4,operation = Individual.crossover(HYGO_params,ind1,ind2)

    ind5,instructions = Individual.mutate(HYGO_params,ind1)

    print(ind1.chromosome,'\n',chromosome_to_params(HYGO_params,ind1.chromosome))
    print(ind2.chromosome,'\n',chromosome_to_params(HYGO_params,ind2.chromosome))
    print(ind3.chromosome,'\n',chromosome_to_params(HYGO_params,ind3.chromosome))
    print(ind4.chromosome,'\n',chromosome_to_params(HYGO_params,ind4.chromosome))
    print(operation)
    print(ind5.chromosome,'\n',chromosome_to_params(HYGO_params,ind5.chromosome))
    print(instructions)
    
    ind = Individual()
    chrom = ind.create_chromosome(HYGO_params)
    chrom = ind.eliminate_introns(HYGO_params,chrom)
    law = ind.chromosome_to_control_law(HYGO_params,chrom)
    
    print(law)
    