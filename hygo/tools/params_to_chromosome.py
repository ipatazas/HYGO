import numpy as np

def params_to_chromosome(HYGO_params,params):
    """
    Converts a list of parameters to a binary chromosome based on the specified number of bits for each parameter.

    Parameters:
        - HYGO_params: A object containing genetic algorithm parameters.
        - params: A list of numerical parameters to be converted to a binary chromosome.

    Returns:
        - list: A binary chromosome representing the input parameters.

    This function takes a list of parameters and converts them to a binary chromosome based on the specified number of
    bits for each parameter in 'HYGO_params'. The conversion is performed using a linear scaling to map parameter values
    within their respective ranges to binary representations.

    Example:
    ```
    # Create a GAParameters object specifying the number of bits for each parameter
    HYGO_params = GAParameters(Nb_bits=[8, 10], params_range=[(0, 10), (-5, 5)])

    # Define a list of parameters to be converted
    input_params = [7.5, -2]

    # Convert parameters to a binary chromosome
    binary_chromosome = params_to_chromosome(HYGO_params, input_params)
    ```

    In the example above, the function converts the input parameters [7.5, -2] to a binary chromosome using the
    specified GAParameters object.
    """
    
    Nb_bits = HYGO_params.Nb_bits
    chromosome=[]
    if type(Nb_bits)!=int:
        if len(Nb_bits)>1:
            for i,param in enumerate(params):
                # Get the current number of bits
                bits = Nb_bits[i]
                
                # Obtain the limits for the current parameter
                xmin = HYGO_params.params_range[i][0]
                xmax = HYGO_params.params_range[i][1]

                # Scale the parameter value to the range [0, 2^bits - 1]
                alpha = (param - xmin)/(xmax - xmin)
                
                # Convert it into bit notation
                x = alpha*(2**bits-1)
                c = bin(int(x))
                c = c[2:]
                
                # Add the bit representation to the chromosome
                idx=0
                for i in range(bits):
                    if i<(bits-len(c)):
                        chromosome.append(0)
                    else:
                        chromosome.append(int(c[idx]))
                        idx+=1
        else:
            for i,param in enumerate(params):
                # Get the current number of bits
                bits = Nb_bits[0]
                
                # Obtain the limits for the current parameter
                xmax = HYGO_params.params_range[i][1]
                xmin = HYGO_params.params_range[i][0]

                # Scale the parameter value to the range [0, 2^bits - 1]
                alpha = (param - xmin)/(xmax - xmin)
                
                # Convert it into bit notation
                x = alpha*(2**bits-1)
                c = bin(int(x))
                c = c[2:]
                
                # Add the bit representation to the chromosome
                idx=0
                for i in range(bits):
                    if i<(bits-len(c)):
                        chromosome.append(0)
                    else:
                        chromosome.append(int(c[idx]))
                        idx+=1
    else:
        for i,param in enumerate(params):
            # Get the current number of bits
            bits = Nb_bits
            
            # Obtain the limits for the current parameter
            xmax = HYGO_params.params_range[i][1]
            xmin = HYGO_params.params_range[i][0]

            # Scale the parameter value to the range [0, 2^bits - 1]
            alpha = (param - xmin)/(xmax - xmin)
            
            # Convert it into bit notation
            x = alpha*(2**bits-1)
            c = bin(int(x))
            c = c[2:]
            
            # Add the bit representation to the chromosome
            idx=0
            for i in range(bits):
                if i<(bits-len(c)):
                    chromosome.append(0)
                else:
                    chromosome.append(int(c[idx]))
                    idx+=1

    return chromosome