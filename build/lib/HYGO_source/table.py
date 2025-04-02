__author__ = 'Isaac Robledo Martín'
import numpy as np

class Table():

    def __init__(self):
        '''
        Initialize the Table class.

        Attributes:
            - latest_indiv (int): The index of the latest individual added.
            - individuals (list): List to store individual objects.
            - hashlist (list): List to store hash values of individuals.
            - costs (list): List to store costs associated with individuals.

        Methods:
            - add_individual(HYGO_params, ind): adds an individual to the saved data.
            - find_indiv(ind): checks if the introducen individual is already in the table.
                If it is, it returns the index of the individual, if it does not, a -1.
            - give_best(n): gives the indexes and costs of the best n individuals in the table.
            - remove_individual(idx): removes the individual of index idx from the table.
        '''
        self.latest_indiv = 0
        self.individuals = []
        self.hashlist = []
        self.costs = []

    def add_individual(self, HYGO_params, ind):
        '''
        Add an individual to the table or update its information.

        Parameters:
            - HYGO_params (object): An object containing parameters for the Genetic Algorithm.
            - ind (Individual): The individual to add or update.

        Returns:
            - int: Index of the added/updated individual
            - bool: Boolean indicating if the individual is new or not.

        '''
        # Check if the individual is already in the table
        idx = self.find_indiv(ind)

        if idx == -1:
            # If the individual is not in the table, add it
            idx = self.latest_indiv
            self.latest_indiv += 1

            # Update table information
            self.individuals.append(ind)
            self.hashlist.append(ind.hash)
            self.costs.append(-1)

            return idx, False
        else:
            # If the individual is already in the table, update information
            if not HYGO_params.remove_duplicates:
                ind.ocurrences += 1
            return int(idx), True
    
    def find_indiv(self, ind):
        '''
        Find the index of an individual in the table based on its hash value. If it does
            not exist it returns a -1

        Parameters:
            - ind (Individual): The individual to search for.

        Returns:
            - int: Index of the individual in the table. Returns -1 if not found.

        '''
        for i, hash_val in enumerate(self.hashlist):
            if ind.hash == hash_val:
                return i
            
        return -1
    
    def give_best(self, n):
        '''
        Retrieve the indices and costs of the best n individuals in the table.

        Parameters:
            - n (int): The number of best individuals to retrieve.

        Returns:
            - list: Indices of the best n individuals.
            - list: Costs of the best n individuals.

        '''
        J = np.array(self.costs)

        ordered_idx = np.argsort(J)

        J_ordered = J[ordered_idx[0:n]]
        idx_ordered = ordered_idx[0:n]

        return idx_ordered, J_ordered

    def remove_individual(self, idx):
        '''
        Remove an individual from the table based on its index.

        Parameters:
            - idx (int): Index of the individual to remove.

        '''
        # Remove individual information from the table
        self.individuals.pop(idx)
        self.hashlist.pop(idx)
        self.costs.pop(idx)

        self.latest_indiv -= 1
