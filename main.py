from HYGO.HYGO import HYGO
from Plant.CheckParams import Parameters
import os
import matplotlib.pyplot as plt

HYGO_params = Parameters()
#Import the results

PATH = 'results/output/'

if not os.path.isdir(PATH):
    os.mkdir(PATH)

#Create the object
result = HYGO(HYGO_params,output_path = PATH)
# result = HYGO.load_security(PATH)
#Run the genetic algorithm
result.help()

# result.save()
# result.save(specific_save=True)

# result.convergence(save=PATH, show=False)
# result.plot_individual_ancestry(200,'output/prueba.png')
# result.plot_dimensionality_reduction('output/dim_red.png')

print('done')