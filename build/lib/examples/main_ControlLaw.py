from HYGO_source.HYGO import HYGO
from examples.parameters_control_law import Parameters
import os

HYGO_params = Parameters()
#Import the results

PATH = 'Results_HYGO_Landau_3/'

if not os.path.isdir(PATH):
    os.mkdir(PATH)

#Create the object
result = HYGO(HYGO_params,output_path = PATH)

result.go()

result.save()
result.save(specific_save=True)

#result = HYGO.load('output.obj')

result.convergence()

'''result.convergence()

result.plot_gens([0,1,2,3,4])'''

'''for i,ind in enumerate(result.table.individuals):
    if abs(ind.parameters[0])>1 or abs(ind.parameters[1])>1:
        print(i,ind.cost)
'''

'''for i,ind in enumerate(result.table.individuals):
    print(i,ind.reconstruction_time)
'''

print('done')