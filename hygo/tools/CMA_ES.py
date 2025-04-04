import numpy as np
from ..individual import *

class CMA_ES:
    """
    The CMA-ES represents the implementation of the CMA-ES optimization algorithm.

    Attributes:
        cma_es_state (str): Current state of the simplex optimization process.
        exploitation (bool): Flag indicating whether exploitation is enabled.
        Lambda (int): number of offspring per loop
        mu (int): number of selected offspring
        w (np.array): weights of the selected individuals
        mueff (float): effective mu
        cc (float): evolution path learning rate
        cs (float): standard variation learning rate
        c1 (float): Rank-1 update learning rate
        cmu (float): mean learning rate
        ds (float): damping coefficient for the std
        sigma (list[float]): list of std of each loop
        pc (list[np.array]): list of the path of each loop
        ps (list[float]): list of the evolution path of each loop
        B (np.array): matrix containing the eigenvectors of the g gen
        D (np.array): matrix containing the eigenvalues of the g gen
        C (list(np.array)): list containing the covariance matrix of each loop
        E (float): expectation of euclidean mean
        yw (list[float]): list of std of each loop
        y (list[float]): list of std of each loop
        x (list[float]): list of std of each loop
        m (list[float]): list of std of each loop

    Methods:
        __init__: Initializes the Simplex object.
        
    """

    def __init__(self):
        """
        Initializes the Simplex object.
        """
        self.cma_es_state = None
        pass

    def CMA_ES_exploitation(self,HYGO_params,HYGO_table,path):
        """
        Executes the exploitation phase using the CMA-ES method.

        Args:
            HYGO_params (object): An object containing genetic algorithm parameters.
            HYGO_table (object): The  table containing information about individuals.
            path (str): The path where the results are stored.

        Returns:
            Tuple[PopulationTable, bool, bool]: A tuple containing the updated population table, a flag indicating whether the maximum number of individuals is reached, and a flag indicating convergence.
        """
        if HYGO_params.verbose:
            print('---> CMA-ES method selected for generation '+str(self.generation))

        # Check to see if a security backup is being computed
        restart = False

        # Check if the CMA-ES was already initialized
        if self.cma_es_state:
            restart = True
            id = len(self.C)
            if self.cma_es_state == 'Loop ' + str(id) + 'started':
                pickup = 1
                i = id
            elif self.cma_es_state == 'Loop ' + str(id) + 'sampled':
                pickup = 2
                i = id
            elif self.cma_es_state == 'Loop ' + str(id) + 'updated':
                pickup = 3
                i = id
            elif self.cma_es_state == 'Loop ' + str(id) + 'finished':
                pickup = 0
                i = id+1
        else:
            # Initialice
            self.initialize_CMA(HYGO_params,HYGO_table)
            i = 1

        # Initialize convergence
        convergence=False
        checker = True

        # Perfome the evolution and evaluation loop
        while i<=HYGO_params.CMA_gens and checker:
            if HYGO_params.verbose:
                print('-----------Starting loop '+str(i)+' -----------')

            if not restart:
                # Change the state
                self.cma_es_state = 'Loop ' + str(i) + 'started'

                # Fill the new loop places
                self.x.append(None)
                self.y.append(None)
                self.m.append(None)
                self.ps.append(None)
                self.sigma.append(None)
                self.pc.append(None)
                self.C.append(None)

                # Generate new individuals and evaluate them
                HYGO_table,checker = self.sample(HYGO_params,HYGO_table,path,i)

                self.cma_es_state = 'Loop ' + str(i) + 'sampled'

                # Check convergence
                if HYGO_params.check_convergence:
                    convergence = self.check_convergence(HYGO_params,HYGO_table)

                if not convergence:
                    # Update the quantities
                    self.update_cuantities(HYGO_params,i)

                    self.cma_es_state = 'Loop ' + str(i) + 'updated'

                    # Update the covariance
                    self.update_covariance(HYGO_params,i)

                    # Change the state
                    self.cma_es_state = 'Loop ' + str(i) + 'finished'
            else:
                if pickup==0:
                    # Change the state
                    self.cma_es_state = 'Loop ' + str(i) + 'started'

                    # Fill the new loop places
                    self.x.append(None)
                    self.y.append(None)
                    self.m.append(None)
                    self.ps.append(None)
                    self.sigma.append(None)
                    self.pc.append(None)
                    self.C.append(None)

                if pickup<=1:
                    # Generate new individuals and evaluate them
                    HYGO_table,checker = self.sample(HYGO_params,HYGO_table,path,i)

                    self.cma_es_state = 'Loop ' + str(i) + 'sampled'
                    # Check convergence
                    if HYGO_params.check_convergence:
                        convergence = self.check_convergence(HYGO_params,HYGO_table)

                if pickup<=2 and not convergence:
                    # Update the quantities
                    self.update_cuantities(HYGO_params,i)

                    self.cma_es_state = 'Loop ' + str(i) + 'updated'

                if pickup<=3 and not convergence:
                    # Update the covariance
                    self.update_covariance(HYGO_params,i)

                    # Change the state
                    self.cma_es_state = 'Loop ' + str(i) + 'finished'
                
                # So next loop is normal
                restart = False
            
            # Update loop number
            i += 1
        
        # Update the state
        self.cma_es_state = 'CMA-ES done'

        return HYGO_table,checker,convergence

    def initialize_CMA(self,HYGO_params,HYGO_table):
        """
        Initializes the parameters for the CMA-ES algorithm.

        Args:
            - HYGO_params (object): Object containing genetic algorithm parameters.
            - HYGO_table (object): The table containing information about individuals.

        Raises:
            ValueError: If the provided exploitation parameter is not a boolean or a list of booleans.

        Returns:
            None
        """

        if HYGO_params.verbose:
            print('    -Initialization')

        # Set the exploitation flag based on the provided parameter
        if type(HYGO_params.exploitation)==list:
            self.exploitation = HYGO_params.exploitation[self.generation-1]
        elif type(HYGO_params.exploitation)==bool:
            self.exploitation = HYGO_params.exploitation
        else:
            raise ValueError('The exploitation must be a bool or a list of bools')

        # Get the offspring per generation
        if hasattr(HYGO_params,'CMA_Lambda'):
            self.Lambda = HYGO_params.CMA_Lambda
        else:
            self.Lambda = int(4+np.floor(3+np.log(HYGO_params.N_params)))

        # Get the reduced number of individuals
        if hasattr(HYGO_params,'CMA_mu'):
            self.mu = int(np.floor(HYGO_params.CMA_mu))
        else:
            self.mu = int(np.floor(self.Lambda/2))

        # Get the weights
        if hasattr(HYGO_params,'CMA_weights'):
            if type(HYGO_params.CMA_weights)!=list:
                raise ValueError('The weights must be introduced as a list')
            self.w = np.array(HYGO_params.CMA_weights)
        else:
            self.w = np.log(self.mu + 0.5) - np.log(np.linspace(1,self.mu,self.mu))
            self.w = self.w/np.sum(self.w) # Normalization

        # Compute the effective mu
        self.mueff = np.power(np.sum(self.w),2)/np.sum(np.power(self.w,2))

        # Get the cc
        if hasattr(HYGO_params,'CMA_cc'):
            self.cc = HYGO_params.CMA_cc
        else:
            self.cc = (4+self.mueff/HYGO_params.N_params)/(HYGO_params.N_params+4 +2*self.mueff/HYGO_params.N_params)
        
        # Get the cs
        if hasattr(HYGO_params,'CMA_cs'):
            self.cs = HYGO_params.CMA_cs
        else:
            self.cs = (self.mueff + 2)/(HYGO_params.N_params + self.mueff + 5)
        
        # Get the c1
        if hasattr(HYGO_params,'CMA_c1'):
            self.c1 = HYGO_params.CMA_c1
        else:
            self.c1 = (2)/((HYGO_params.N_params+1.3)**2 + self.mueff)

        # Get the cmu
        if hasattr(HYGO_params,'CMA_cmu'):
            self.cmu = HYGO_params.CMA_cmu
        else:
            self.cmu = np.min([1-self.c1,2*(self.mueff-2+1/self.mueff)/((HYGO_params.N_params+2)**2 + 2*self.mueff/2)])

        # Get the cc
        if hasattr(HYGO_params,'CMA_ds'):
            self.ds = HYGO_params.CMA_ds
        else:
            if ((self.mueff-1)/(HYGO_params.N_params+1) - 1)>0:
                self.ds = 1 + 2*np.max([0.0,np.sqrt((self.mueff-1)/(HYGO_params.N_params+1) - 1)]) + self.cs
            else:
                self.ds = 1+self.cs

        # Initialize the covariance an the expectation of Euclidean norm
        if HYGO_params.CMA_Pool == 'Population':
            # Get the individuals parameters and cost
            params = []
            costs = []
            for i in range(self.data.shape[0]):
                idx = int(self.data.loc[i,'Individuals'])
                params.append(HYGO_table.individuals[idx].parameters) 
                costs.append(HYGO_table.individuals[idx].cost)   
        elif HYGO_params.CMA_Pool == 'All':
            # Get the individuals parameters and cost
            params = []
            costs = []
            for idx in range(len(HYGO_table.individuals)):
                params.append(HYGO_table.individuals[idx].parameters)
                costs.append(HYGO_table.individuals[idx].cost)  

        # Analogous to x0
        params = np.array(params)
        # Obtain the order according to cost
        idx_order = np.argsort(costs)

        # Extract the valid x0
        x0 = np.zeros((HYGO_params.N_params,self.Lambda))
        for i in range(self.Lambda):
            x0[:,i] = np.array(params[idx_order[i]])

        # Obtain m0 and std0
        self.m = [np.mean(x0[:,1:self.mu],axis=1).reshape(HYGO_params.N_params,1)]
        self.sigma = [np.std(x0[:,1:self.mu])]
        #self.sigma = [1]

        # Compute the covariance matrix
        self.C = [np.cov(x0[:,1:self.mu])]
        # Enforce symmetry
        self.C[0] = np.triu(self.C[0]) + np.triu(self.C[0],k=0).T

        # Compute the B and C matrices
        eigvals, self.B = np.linalg.eig(self.C[0])
        self.D = np.diag(np.sqrt(eigvals))

        self.E = np.sqrt(HYGO_params.N_params)*(1 - 1/(4*HYGO_params.N_params) + 1/(21*HYGO_params.N_params**2))

        # Obtain y0
        y0 = np.zeros((HYGO_params.N_params,self.Lambda))
        for i in range(self.Lambda):
            pars = np.array(params[idx_order[i]])
            y0[:,i] = ((pars.reshape(self.m[0].shape)-self.m[0])/self.sigma[0]).reshape(y0[:,i].shape)

        # Compute yw
        self.yw = np.zeros((HYGO_params.N_params,1))
        for i in range(self.mu):
            self.yw[:,0] = self.yw[:,0] + self.w[i]*y0[:,i]

        # Initialize ps
        term = self.cs*(2-self.cs)*self.mueff
        mat = np.power(np.linalg.inv(self.C[0]),0.5)
        if term>=0 and not np.isnan(mat).any():
            self.ps = [np.sqrt(term)*np.matmul(mat,self.yw)]
        else:
            self.ps = [np.zeros((HYGO_params.N_params,1))]

        # Initialize pc
        hs = np.linalg.norm(self.ps[0])/np.sqrt(1 - np.power(1-self.cs,2)) 
        if hs < (1.4+2/(HYGO_params.N_params+1)) and not np.isnan(hs):
            self.pc = [hs*np.sqrt(self.cc*(2-self.cc)*self.mueff)*self.yw]
        else:
            self.pc = [np.zeros((HYGO_params.N_params,1))]

        # Set the initial values of the variables to 0
        self.y = [np.zeros((self.Lambda,1))]
        self.x = [np.zeros((self.Lambda,1))]

        # Update the state
        self.cma_es_state = 'Exploitation initialized'

    def sample(self,HYGO_params,HYGO_table,path,g):
        '''
        This function computes the zk, yk, xk and yw values for generation g+1:

        Arguments:
            HYGO_params (object): An object containing genetic algorithm parameters.
            HYGO_table (object): Table containing the population of individuals.
            path (str): The path to save additional information about the operations performed.
            g (int): new generation number (g+1)

        Computes:
            yk(np.matrix): matrix containing the new yk^g+1 ordered based in cost
            xk(np.matrix): matrix containing the new xk^g+1 ordered based in cost
        
        Returns:
            Tuple[PopulationTable, bool]: A tuple containing the updated population table, a flag indicating whether the maximum number of individuals is reached
        '''

        yk = np.zeros((HYGO_params.N_params,self.Lambda))
        xk = np.zeros((HYGO_params.N_params,self.Lambda))

        # Create a list with the individuals indexes
        idx_indivs = []

        if HYGO_params.verbose:
            print('    -Sampling '+str(self.Lambda)+ ' individuals')

        for i in range(self.Lambda):
            # Generate an individual until it does not exist
            checker = True
            counter = 1
            if HYGO_params.verbose:
                print('Generating individual '+str(i+1) + '/'+str(self.Lambda))
            while checker:
                # Obtain a new zk
                zk = np.random.randn(HYGO_params.N_params,1)
                # Compute the new parameters
                yk[:,i] = np.matmul(np.matmul(self.B,self.D),zk).reshape(yk[:,i].shape)
                xk[:,i] = (self.m[g-1] + self.sigma[g-1]*yk[:,i].reshape(self.m[g-1].shape)).reshape(xk[:,i].shape)
                #xk[:,i] = (self.m[g-1] + yk[:,i].reshape(self.m[g-1].shape)).reshape(xk[:,i].shape)

                if np.isnan(xk[:,i]).any():
                    print('aaa')
                # Ensure the new parameters are inside the bounds
                xk[:,i] = np.array(self.check_params(xk[:,i].tolist(),HYGO_params))
                # Create a new individual with the reflected parameters
                new_ind = Individual()
                new_ind.create(HYGO_params=HYGO_params,params=xk[:,i].tolist())
                # Add individual to the table
                [idx,exists] = HYGO_table.add_individual(HYGO_params=HYGO_params,ind=new_ind)

                # Assume that the individual is valid
                valid = True

                # Check if there is a custom validity function
                if hasattr(HYGO_params,'validity') and callable(HYGO_params.validity):
                    custom_valid = HYGO_params.validity(new_ind.parameters)
                    valid = valid and custom_valid

                # Remove the individual if not valid
                if (not valid and not exists) and counter<HYGO_params.MaxTries and HYGO_params.remove_duplicates:
                    HYGO_table.remove_individual(int(idx))
                
                checker = HYGO_params.remove_duplicates and counter<HYGO_params.MaxTries and (exists or not valid)

                counter+=1

            # Update the yk with the adjusted xk
            yk[:,i] = ((xk[:,i].reshape(self.m[g-1].shape)-self.m[g-1])/self.sigma[g-1]).reshape(yk[:,i].shape)
            #yk[:,i] = ((xk[:,i].reshape(self.m[g-1].shape)-self.m[g-1])).reshape(yk[:,i].shape)

            # Update the index list
            idx_indivs.append(idx)

            # Get the index of the individual in the population
            nind = self.data.shape[0]

            # Update the population data with information about the mutated individual
            if exists:
                self.data.loc[nind,['Costs']] = HYGO_table.individuals[idx].cost

            # If it is not valid, assign a badvalue so it is not evaluated
            if not valid and not exists:
                self.data.loc[[nind],['Costs']] = HYGO_params.badvalue
                HYGO_table.individuals[idx].cost = HYGO_params.badvalue

            self.data.loc[nind,'Individuals'] = idx
            self.data.loc[nind,('Parents','first')] = -1
            self.data.loc[nind,('Parents','second')] = -1
            self.data.loc[nind,('Operation','Type')] = 'CMA-ES'
            self.data.loc[nind,('Operation','Point_parts')] = str('None')

            # Update indexes that have to be evaluated
            for rep in range(HYGO_params.repetitions):
                self.idx_to_evaluate[rep] = self.idx_to_evaluate[rep] + [nind]
        
        '''import matplotlib.pyplot as plt
        Cov = self.C[g-1]
        theta = np.linspace(0, 2*np.pi, 1000)
        eigenvalues, eigenvectors = np.linalg.eig(Cov)
        ellipsis = (np.sqrt(eigenvalues[None,:]) * eigenvectors) @ [np.sin(theta), np.cos(theta)]

        plt.plot(ellipsis[0,:], ellipsis[1,:])
        plt.scatter(yk[0,:],yk[1,:])

        plt.plot(ellipsis[0,:]+self.m[g-1][0], ellipsis[1,:]+self.m[g-1][1])
        plt.scatter(xk[0,:],xk[1,:])
        plt.show()'''

        # Evaluate the individuals
        HYGO_table,checker = self.evaluate_population(copy.deepcopy(self.idx_to_evaluate),HYGO_params,HYGO_table,path,True)

        # Obtain the costs
        costs = []
        for idx in idx_indivs:
            costs.append(HYGO_table.individuals[idx].cost)

        # Order the indexes according to cost
        idx_ordered = np.argsort(np.array(costs))

        idx_indivs = np.array(idx_indivs)[idx_ordered]
        xk = xk[:,idx_ordered]
        yk = yk[:,idx_ordered]

        # Compute yw
        self.yw = np.zeros((HYGO_params.N_params))
        for i in range(self.mu):
            self.yw += self.w[i]*yk[:,i]

        self.x[g] = xk
        self.y[g] = yk

        return HYGO_table,checker
    
    def update_cuantities(self,HYGO_params,g):
        '''
        This function computes the mean, std, pc and ps for generation g+1:

        Arguments:
            HYGO_params (object): An object containing genetic algorithm parameters.
            g (int): new generation number (equivalent to g+1)

        Computes:
            m(g): mean of new generation
            ps(g): path of new generation
            sigma(g): std of new generation
            pc(g): evolution path of new generation
            '''
        if HYGO_params.verbose:
            print('    -Updating cuantities')
        # Compute new mean
        self.m[g] = self.m[g-1] + (self.cmu*self.sigma[g-1]*self.yw).reshape(self.m[g-1].shape)

        # Compute new path
        term = self.cs*(2-self.cs)*self.mueff
        mat = np.power(np.linalg.inv(self.C[0]),0.5)
        if term>=0 and not np.isnan(mat).any():
            self.ps[g] = (1-self.cs)*self.ps[g-1] + np.sqrt(term)*np.matmul(mat,self.yw).reshape(self.ps[g-1].shape)
        else:
            self.ps[g] = (1-self.cs)*self.ps[g-1]

        # Compute new std
        print(np.exp((self.cs/self.ds)*(np.linalg.norm(self.ps[g])/self.E)-1))
        self.sigma[g] = self.sigma[g-1]*np.exp((self.cs/self.ds)*(np.linalg.norm(self.ps[g])/self.E)-1)

        # Compute new evolution path
        hs = np.linalg.norm(self.ps[g])/np.sqrt(1 - np.power(1-self.cs,2*(g+2))) 
        if hs < (1.4+2/(HYGO_params.N_params+1)) and not np.isnan(hs):
            self.pc[g] = hs*np.sqrt(self.cc*(2-self.cc)*self.mueff)*self.yw
        else:
            self.pc[g] = np.zeros((HYGO_params.N_params,1))
    
    def update_covariance(self,HYGO_params,g):
        '''
        This function computes the new covariance matrix for generation g+1:

        Arguments:
        HYGO_params (object): An object containing genetic algorithm parameters.
            g (int): new generation number (equivalent to g+1)

        Computes:
            C(g): covariance matrix of generation g
        '''
        if HYGO_params.verbose:
            print('    -Updating covariance')
        # Compute the new heaviside function
        hs = np.linalg.norm(self.ps[g])/np.sqrt(1 - np.power(1-self.cs,2*(g+2))) 
        if not np.isnan(hs):
            deltah = (1-hs)*self.cc*(2-self.cc)
        else:
            deltah = self.cc*(2-self.cc)

        # Compute sums
        sum1 = self.cmu*np.sum(self.w)
        sum2 = 0
        for i in range(self.mu):
            sum2 += self.cmu*self.w[i]*np.matmul(self.y[g][:,i],self.y[g][:,i].T)

        # Compute the new covariance
        C = (1+self.c1*deltah-self.c1-sum1)*self.C[g-1] + self.c1*np.matmul(self.pc[g],self.pc[g].T) + sum2

        # Enorce symmetry
        C = np.triu(C) + np.triu(C,k=0).T

        # Compute the B and C matrices
        eigvals, eigenvects = np.linalg.eig(C)

        if not np.isnan(np.sqrt(eigvals)).any():
            self.C[g] = C
            self.B = eigenvects
            self.D = np.diag(np.sqrt(eigvals))
        else:
            self.C[g] = self.C[g-1].copy()