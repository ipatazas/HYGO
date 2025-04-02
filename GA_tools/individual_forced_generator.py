import numpy as np
import scipy.io

def individual_forced_generator(HYGO_params):
    
    custom_params = []
    
    if HYGO_params.name=='ahby':
        dat_ga = scipy.io.loadmat('dat_GA.mat')['data_GA']
        custom_params = dat_ga[:11,:-1]
        custom_params=custom_params.tolist()

    if HYGO_params.name == 'Rosenbrock':
        custom_params = np.multiply((np.random.rand(HYGO_params.pop_size,2)-8),np.where(np.random.randint(0,2,(HYGO_params.pop_size,2))<=0,-1,1)).tolist()

    '''if HYGO_params.name == 'Example':
        custom_params.append([1,1,1,1,1,1])
    elif HYGO_params.name == 'Rosenbrock':
        custom_params = [[1.1,1]]
    
    elif HYGO_params.name == 'Goldstein':
        posx = np.linspace(HYGO_params.params_range[0][0],HYGO_params.params_range[0][1],2**HYGO_params.Nb_bits[0])
        posy = np.linspace(HYGO_params.params_range[1][0],HYGO_params.params_range[1][1],2**HYGO_params.Nb_bits[1])
        posy = posy[posy>1]

        for _ in range(HYGO_params.pop_size):
            custom_params.append([posx[np.random.randint(len(posx))],posy[np.random.randint(len(posy))]])
    elif HYGO_params.name == 'Ackley' or HYGO_params.name == 'Rastrigin':
        pos = np.linspace(HYGO_params.params_range[0][0],HYGO_params.params_range[0][1],2**HYGO_params.Nb_bits)

        np.random.seed(1)
        custom_params=pos[np.random.randint(0,pos.size,(HYGO_params.pop_size,HYGO_params.N_params))]
        custom_params=custom_params.tolist()'''

    return custom_params