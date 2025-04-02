__author__ = 'Isaac Robledo Martín'
import numpy as np

def chromosome_to_params(HYGO_params,chromosome):
    """
    Converts a binary chromosome to a list of parameter values using the provided genetic algorithm parameters.

    Parameters:
        - HYGO_params: An object containing genetic algorithm parameters, including the number of bits per parameter (Nb_bits),
                    the number of parameters (N_params), and the parameter value ranges.
        - chromosome: A binary chromosome represented as a list or array of bits.

    Returns:
        - list: A list of parameter values corresponding to the given chromosome.

    This function interprets the binary representation of parameters in the chromosome and converts them to their
    corresponding real values based on the specified number of bits per parameter. The conversion involves scaling the
    binary values to the specified parameter range.

    Example:
    ```
    binary_chromosome = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    parameters = chromosome_to_params(HYGO_params, binary_chromosome)
    ```

    In the example above, the binary chromosome is converted to a list of parameter values based on the provided
    genetic algorithm parameters.
    """
    Nb_bits = HYGO_params.Nb_bits # Number of bits assigned to each parameter
    N_params = HYGO_params.N_params # Total number of parameters
    params=[]# List to store the converted parameter values
    
     # Check if the number of bits is a single value or a list for each parameter
    if type(Nb_bits)!=int:
        if len(Nb_bits)>1:
            
            # Keep track of the last index used in the chromosome
            last_idx=0
            for i in range(N_params):
                n_bits=Nb_bits[i] # Number of bits for the current parameter
                base2 = np.power(2.0,-(np.array(range(n_bits))+1)) # Binary base to decimal conversion weights
                bits=chromosome[last_idx:last_idx+n_bits]  # Extract bits for the current parameter
                last_idx+=n_bits # Update the last index for the next parameter
                xtpm = np.sum(np.dot(base2,bits)).tolist() # Convert binary to decimal
                xmin = HYGO_params.params_range[i][0]  # Minimum value for the current parameter
                xmax = HYGO_params.params_range[i][1]  # Maximum value for the current parameter
                params.append(xmin+(xmax-xmin)/(1-2**(-n_bits))*xtpm) # Scale and append the parameter
        else:
            n_bits=Nb_bits[0] # Number of bits for all parameters (assuming uniform bit allocation)
            base2 = np.power(2.0,-(np.array(range(n_bits))+1)) # Binary base to decimal conversion weights
            for i in range(N_params):
                bits=chromosome[n_bits*i:n_bits*(i+1)] # Extract bits for the current parameter
                xtpm = np.sum(np.dot(base2,bits)).tolist() # Convert binary to decimal  
                xmin = HYGO_params.params_range[i][0] # Minimum value for the current parameter
                xmax = HYGO_params.params_range[i][1] # Maximum value for the current parameter
                params.append(xmin+(xmax-xmin)/(1-2**(-n_bits))*xtpm) # Scale and append the parameter  
    else:
        n_bits=Nb_bits # Number of bits for all parameters (assuming uniform bit allocation)
        base2 = np.power(2.0,-(np.array(range(n_bits))+1)) # Binary base to decimal conversion weights
        for i in range(N_params): 
            bits=chromosome[n_bits*i:n_bits*(i+1)] # Extract bits for the current parameter
            xtpm = np.sum(np.dot(base2,bits)).tolist() # Extract bits for the current parameter
            xmin = HYGO_params.params_range[i][0] # Minimum value for the current parameter
            xmax = HYGO_params.params_range[i][1] # Maximum value for the current parameter
            params.append(xmin+(xmax-xmin)/(1-2**(-n_bits))*xtpm) # Scale and append the parameter

    return params