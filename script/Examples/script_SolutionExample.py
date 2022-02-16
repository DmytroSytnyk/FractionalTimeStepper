
from math import *
import numpy as np
import dolfin as dl
from scipy.special import gamma
import matplotlib.pyplot as plt
from time import time
import sys, os

###=============================================
### Import from source

SOURCEPATH = os.path.abspath(__file__)
for i in range(10):
    basename   = os.path.basename(SOURCEPATH)
    SOURCEPATH = os.path.dirname(SOURCEPATH)
    if basename == "script": break
sys.path.append(SOURCEPATH)

from source.Models.DiffusionModel1D import DiffusionModel1D
from source.Models.SimpleScalarModel import SimpleScalarModel
from source.TimeStepper import FracTS_RA, FracTS_L1, FracTS_GJ
from source.MittagLeffler import ml ### Load some Mittag-Leffler function computation code, e.g., see https://github.com/khinsen/mittag-leffler

####################################
#           Parameters
####################################

model_type = 'diffusion1d' # 'scalar', 'diffusion1d'

config = {
    'alpha'         :   0.5,
    'nTimeSteps'    :   100,
    'FinalTime'     :   1,
    'stiff'         :   1,
    'BC'            :   ['Dirichlet', '0'],
    'source'        :   '0',
    'shape'         :   [5000], ### spacial domain discretization
    'nSupportPoints':   100,    ### AAA candidate points
    'tol'           :   1.e-13, ### AAA accuracy
    'verbose'       :   False,
}
scheme = 'RA:mCN'  ### Format "XX:Y..Y"; XX = RA / L1 / GJ; for RA: Y..Y = mIE - modified Implicit Euler (theta=1), mCN - modified Crank-Nicolson
    

if model_type == 'scalar':
    config['IC'] = 1
    mdl = SimpleScalarModel(**config)
elif model_type == 'diffusion1d':
    config['IC'] = 'sin(pi*x[0])'
    mdl = DiffusionModel1D(**config)
else:
    raise Exception('Unknown model!')

### initial state
u_init = mdl.ArrInitSol

####################################
### Definitions of the norms

def normH(u):
    return mdl.norm(u)

def normT_weighted(u):
    t = np.arange(len(u))
    return np.linalg.norm(u * t**(1-alpha), ord=2)

def normT(u):
    # return np.linalg.norm(u, ord=2)
    # return np.linalg.norm(u, ord=np.Inf)
    # return np.abs(u[-1])
    return normT_weighted(u)


####################################
#       Computational body
####################################

if scheme[:2] == 'L1':
    pb = FracTS_L1(**config, Model=mdl)
elif scheme[:2] == 'RA':
    pb = FracTS_RA(**config, Model=mdl)
elif scheme[:2] == 'GJ':
    pb = FracTS_GJ(**config, Model=mdl)

### Info
print('-------------------------')
print('scheme : {0}'.format(pb.Scheme))
print('alpha  : {0}'.format(pb.alpha))
print('dt     : {0}'.format(pb.dt))

ExactSolution = list(pb(ReferenceSolution=True))
AppxSolution  = pb()  ### Numerical solution: generaor, to get a list, call [pb()]

### Time integration
for i, u in enumerate(AppxSolution):
    u_ref = ExactSolution[i]
    error_i = normH(u-u_ref)
    print(f"Time step {i}: Error = {error_i}")
    ### YOUR CODE: do something with u (solution approximation at t_i)



