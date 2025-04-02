

def round_params(HYGO_params,params):
    """
    Rounds the specified parameters using custom rounding functions if provided in the GA_Parameters object.

    Parameters:
        - HYGO_params: An object containing genetic algorithm parameters.
        - params: A list of numerical parameters to be rounded.

    Returns:
        - list: A list of rounded parameters.

    This function checks if custom rounding functions are specified for each parameter in the Ga_parameters object. If a
    custom rounding function is found, it is applied to the corresponding parameter. The function returns a list of
    rounded parameters.

    Example:
    ```

    # Define a list of parameters to be rounded
    input_params = [7.8, 3.3]

    # Round parameters using custom rounding functions
    rounded_params = round_params(HYGO_params, input_params)
    ```

    In the example above, the function rounds the input parameters [7.8, 3.3] using the specified custom rounding
    functions and returns the rounded parameters.
    """

    if len(HYGO_params.custom_parameters) == len(params):
        for i,param in enumerate(params):
            if callable(HYGO_params.custom_parameters[i]):
                # Apply the custom rounding function to the parameter
                fun = HYGO_params.custom_parameters[i]
                params[i] = fun(param)

    return params