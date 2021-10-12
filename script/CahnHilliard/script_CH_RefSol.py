
from math import *
import numpy as np
import dolfin as dl
from scipy.special import gamma
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

import sys
sys.path.append("/home/khristen/Projects/FractionalTimeSolver/FracTS/code/source/")

from Models.CahnHilliardModel import CahnHilliard, InitialConditions
from FractionalDerivative import FractionalDerivativeL1 as FractionalDerivative



####################################
#           Parameters
####################################


### Chemical potential derivative
# E = 1
# phi1 = lambda x: E*(3 * x)                ### convex part
# phi2 = lambda x: E*(4*x**3 - 6*x*2 - x)   ### concave part
# lmbda = 1.e-3     # surface parameter


config = {
    'alpha'     :   0.7,
    'FinalTime' :   1,
    'nTimeSteps':   int(1.e4),
    'CH_scheme' :   'Split_O2', #'Stab', #'Secant', #'Split_O2',
    'theta'     :   1,
    # 'dt'        :   1.e-4,
    'IC'        :   InitialConditions,
    'source'    :   '0',
    'lmbda'     :   1.e-2,
    'E'         :   1,
    'D'         :   1,
    'S'         :   0,
    # 'phi1'      :   phi1,
    # 'phi2'      :   phi2,
    'Nx'        :   2**6,
    'verbose'   :   False,
}


# alpha = config['alpha']
# T = config['FinalTime']
# N = config['nTimeSteps']
# dt = T/N

ResultsFolder = '/home/khristen/Projects/FractionalTimeSolver/FracTS/code/data/CahnHilliard/Reference_Solution/'
config['OutputFile'] = ResultsFolder + 'ReferenceSolution.pvd'
# config['OutputFile'] = 'ReferenceSolution_a={0:.3f}_T={1:f}_dt={2:f}.pvd'.format(alpha, T, dt)


FracDer = FractionalDerivative(**config)
problem = CahnHilliard(**config, FD=FracDer)

dt = problem.TS.dt
N  = problem.TS.nTimeSteps
print('dt=', dt)

energy = [problem.Energy()]
t_list = [0]

avgStepRuntime = 0
time_start_loc = time()
time_start = time()

for i, u in tqdm(enumerate(problem()), total=N):
    avgStepRuntime += time()-time_start_loc
    np.save(ResultsFolder + 'ReferenceSolution_i={0:d}'.format(i), u)
    energy.append(problem.Energy())
    t_list.append(problem.TS.CurrTime)
    time_start_loc = time()

print('Total runtime: ', time()-time_start)

avgStepRuntime /= N
print('Avg step runtime: ', avgStepRuntime)


plt.plot(t_list,energy)
plt.show()



