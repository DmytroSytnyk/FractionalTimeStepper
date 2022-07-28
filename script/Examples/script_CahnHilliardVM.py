
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

from source.Models.CahnHilliardVMModel import CahnHilliardVM, InitialConditions
from source.TimeStepper.FractionalDerivative import FractionalDerivativeRA

####################################
#           Cofiguration
####################################

config = {
    'alpha'         :   0.5,
    'FinalTime'     :   4,
    'nTimeSteps'    :   1000,
    'IC'            :   InitialConditions,
    'eps'           :   1.e-1,  ###  surface parameter
    'mobility'      :   1.e0,
    'Nx'            :   2**5,
    'verbose'       :   False,
### RA parameters:
    'nSupportPoints':   100,    ### AAA candidate points
    'tol'           :   1.e-13, ### AAA accuracy
}
    
FracDerRA = FractionalDerivativeRA(**config)
pb = CahnHilliardVM(**config, FD=FracDerRA)

pb.TS.RA.plot() ### verify the rational approximation

####################################
#       Computational body
####################################

Solution = pb()  ### Numerical solution: generaor, to get a list, call [pb()]

m  = np.zeros(pb.TS.nTimeSteps+1) # mass
E0 = np.zeros(pb.TS.nTimeSteps+1) # energy (classical)
Em = np.zeros(pb.TS.nTimeSteps+1) # modified energy (see the paper)
Eh = np.zeros(pb.TS.nTimeSteps+1) # history energy (see the paper)

### Time integration
for i, u in tqdm(enumerate(Solution), total=pb.TS.nTimeSteps+1):
    m[i]  = pb.mass
    E0[i] = pb.Energy
    Em[i] = pb.EnergyModified
    Eh[i] = pb.HistoryEnergy
    # print(f"Time step {i}:")
    # print(f"    mass   = {m[i]}")
    ### YOUR CODE: do something with u (solution approximation at t_i)

####################################
#       Plots
####################################

plt.figure("Mass evolution")
plt.plot(m)
plt.title("Mass evolution")
plt.xlabel("time")
plt.ylabel("mass")

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

plt.show()
