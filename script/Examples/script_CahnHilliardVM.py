
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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

from source.Models.CahnHilliardVMModel import CahnHilliardVM, InitialConditions, IC_TwoBubbles, IC_BubbleSoup   
from source.TimeStepper.FractionalDerivative import FractionalDerivativeRA

####################################
#           Configuration
####################################

config = {
### Model parameters:
    'alpha'         :   0.5,
    'FinalTime'     :   4,
    'scheme'        :   'RA:mCN',
### Format "XX:Y..Y"; XX = RA / L1 / GJ; 
##  for RA: Y..Y = mIE - modified Implicit Euler (theta=1), mCN - modified Crank-Nicolson
    'nTimeSteps'    :   2000,
    'eps'           :   1e-2,                               ### surface parameter
    'mobility'      :   lambda x: (1 - x**2)**2,            ### mobility
    'Phi'           :   lambda x: 0.5 * (1-x**2)**2,        ### potential
    'phi1'          :   lambda x: 0.5 * (8 * x),            ### convex part the potential derivative
    'phi2'          :   lambda x: 0.5 * (4*x**3 - 12*x),    ### concave part the potential derivative
    'Nx'            :   2**6,
    'autotune_mu0'  :   True,                               ### Make mu0 to satisfy second equation
    'verbose'       :   True,
### RA parameters:
    'nSupportPoints':   100,    ### AAA candidate points
    'tol'           :   1.e-13, ### AAA accuracy
### Solution export parameter:
    'ExportFolder'  :   './CahnHilliard/IC_BubbleSoup/',
    'export_every'  :   0.1, 
    'Name'          :   'phi',
}
    # 'IC'            :   IC_TwoBubbles,
    # 'IC'            :   InitialConditions,
#### IC_BubbleSoup Initial Condition function generate random bubbles with specified parameters
config['IC']           =   IC_BubbleSoup                    ### Initial conditions
# config['ICParameters'] = {'n': 10, 'R': 0.1, 'r': 0.003}    ### Optional parameters to IC
# config['ICParameters'] = {'n': 3, 'eps': config['eps']}    ### Optional parameters to IC
### Working connfigureation sets:

## 3 Bubbles:
# config['ICParameters'] = {'n': 3, 'seed': 0}    ### Optional parameters to IC
# config['ICParameters'] = {'n': 3, 'seed': 1}    ### Optional parameters to IC
# config['ICParameters'] = {'n': 3, 'seed': 42}    ### Optional parameters to IC

## 10 Bubbles:
config['ICParameters'] = {'n': 10, 'eps': 0.04, 'R': 0.2, 'r': 0.005, 'seed': 3}    ### Optional parameters to IC

# config['ICParameters'] = {'n': 10, 'R': 0.1, 'r': 0.003}    ### Optional parameters to IC
config['FilePrefix']   = 'CH_a_{alpha:.3f}_T_{FinalTime:f}_nt_{nTimeSteps:d}_Nx_{Nx:d}_e_{eps:.4f}'.format(**config)
# import pdb; pdb.set_trace()    
# FracDerRA = FractionalDerivativeRA(**config)
# pb = CahnHilliardVM(**config, FD=FracDerRA)
pb = CahnHilliardVM(**config)

# pb.TS.RA.plot() ### verify the rational approximation

####################################
#       Computational body
####################################

Solution = pb()  ### Numerical solution: generator, to get a list, call [pb()]

m  = np.zeros(pb.TS.nTimeSteps+1) # mass
E0 = np.zeros(pb.TS.nTimeSteps+1) # energy (classical)
Em = np.zeros(pb.TS.nTimeSteps+1) # modified energy (see the paper)
Eh = np.zeros(pb.TS.nTimeSteps+1) # history energy (see the paper)

# Save initial value
pb.Exporter.export_iv_vtk()
### Time integration
for i, u in tqdm(enumerate(Solution), total=pb.TS.nTimeSteps+1):
    m[i]  = pb.mass
    E0[i] = pb.Energy
    Em[i] = pb.EnergyModified
    Eh[i] = pb.HistoryEnergy
    print(f"Time step {i}:")
    print(f"    mass   = {m[i]}")
    ### YOUR CODE: do something with u (solution approximation at t_i)
    pb.Exporter.export_step_vtk(**config)

####################################
#       Plots
####################################


plt.figure("Mass evolution")
plt.plot(m)
plt.title("Mass evolution")
plt.xlabel("time")
plt.ylabel("mass")
plt.savefig(os.path.join(config['ExportFolder'],'Mass_'+config['FilePrefix']+'.png'))

plt.figure("Energy evolution")
plt.subplot(121)
plt.plot(E0, "-", label="Energy (std)")
plt.plot(Em, "--", label="Energy (modified)")
plt.legend()
plt.title("Energy evolution")
plt.xlabel("time")
plt.ylabel("energy")

plt.subplot(122)
plt.plot(Eh, "-", label="History energy")
plt.legend()
plt.title("History energy")
plt.xlabel("time")
plt.ylabel("energy")
plt.savefig(os.path.join(config['ExportFolder'],'Energy_'+config['FilePrefix']+'.png'))
plt.show()
