import numpy as np
def findKClosest(parameters,minima,K):
    """
    Finds the indices of the K closest points to a given set of minima in a parameter space.

    Parameters:
        - parameters: A list or array of points in the parameter space.
        - minima: A list or array representing the coordinates of the minima in the parameter space.
        - K: The number of closest points to be found.

    Returns:
        - np.array: An array of indices corresponding to the K closest points in the 'parameters' array.

    This function calculates the Euclidean distance between each point in the 'parameters' array and the provided set
    of minima. It then returns the indices of the K closest points based on the calculated distances.

    Example:
    ```
    parameter_space = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    minima_coordinates = [4, 5, 6]
    K_closest_indices = findKClosest(parameter_space, minima_coordinates, 2)
    ```

    In the example above, the function finds the indices of the 2 closest points in 'parameter_space' to the provided
    set of minima coordinates.
    """

    distances = [] # List to store the calculated distances
    minima=np.array(minima)  # Convert minima to a NumPy array for efficient calculations

    # Calculate the Euclidean distance for each point in the parameter space
    for params in parameters:
        distances.append(np.sqrt(np.sum(np.power(np.array(params)-minima,2))))

    # Get the indices of the K closest points based on the calculated distances
    idx = np.argsort(np.array(distances))

    return idx[:K]


if __name__ == '__main__':

    print(findKClosest([2.3,4.3,6.5,4.7,8.5,4.23,3.45],4,3))