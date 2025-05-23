from hygo import HYGO
from examples.parameters_Rastriguin import Parameters
import os
import matplotlib.pyplot as plt

# Create the Parameters
HYGO_params = Parameters()

# Select the Path
PATH = 'results/'

if not os.path.isdir(PATH):
    os.mkdir(PATH)

#Create the object
result = HYGO(HYGO_params,output_path = PATH)

# Load the security backup if necessary
# result = HYGO.load_security(PATH)

# Get help if necessary
result.help()

#Run the genetic algorithm
result.go()

# Save the information
result.save()
result.save(specific_save=True)

# Plot the results
result.convergence(save=PATH)

print('done')