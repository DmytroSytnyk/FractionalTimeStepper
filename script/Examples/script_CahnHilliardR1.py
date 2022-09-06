
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

from source.Models.CahnHilliardRModel import CahnHilliardR1

####################################
#           Configuration
####################################

config = {
### Model parameters:
    'alpha'         :   0.9,
    'FinalTime'     :   4,
    'scheme'        :   'RA:mCN',
### Format "XX:Y..Y"; XX = RA / L1 / GJ; 
##  for RA: Y..Y = mIE - modified Implicit Euler (theta=1), mCN - modified Crank-Nicolson
    'nTimeSteps'    :   2**(15),
    'eps'           :   0.03,                                 ### surface parameter
    'M'             :   '0.05',                               ### mobility
    'Md'            :   '0.0',                                ### Derivative of mobility
    # 'M'             :   'lambda x: (1 - x**2)**2',             ### mobility
    # 'Md'            :   'lambda x: -4*x*(1 - x**2)',           ### Derivative of mobility
    'Phi'           :   'lambda x: 0.25 * (1-x**2)**2',        ### potential
    'phi1'          :   'lambda x: 0.25 * (8 * x)',            ### convex part the potential derivative
    'phi2'          :   'lambda x: 0.25 * (4*x**3 - 12*x)',    ### concave part the potential derivative
    'Nx'            :   2**6,
    'LinSolve'      :   True,
    'autotune_mu0'  :   False,                                ### Make mu0 to satisfy second equation
    'verbose'       :   True,
### RA parameters:
    'nSupportPoints':   100,    ### AAA candidate points
    'tol'           :   1.e-13, ### AAA accuracy
### Solution export parameter:
    'FilePrefix'    :   '',                                   ### Empty means autogenerated based on config hash
    'export_every'  :   0.01, 
    'Name'          :   'phi',
}
config['IC']           =   'IC_FourBubbles'
#config['ICParameters'] = {'eps': 0.03}    ### Optional parameters to IC
# config['IC']           =   'IC_LinearCombination'
# config['ICParameters'] = {
        # 'n': 2,
        # 'IC1': 'IC_FourBubbles', 'IC1_Parameters': {'eps': 0.03},
        # 'IC2': 'IC_RandomField', 'IC2_Parameters': {'seed': 1, 'eps': 1,'center': 1}, 
        # 'IC_weights': (1,0.1)} 
# config['IC']           =   'IC_FourBubbles'
# config['ICParameters'] = {'eps': 0.03}    ### Optional parameters to IC
# config['IC']           =   'IC_TwoBubbles'
# 'IC'            :   InitialConditions,
#### IC_BubbleSoup Initial Condition function generate random bubbles with specified parameters
# config['IC']           =   'IC_BubbleSoup'                    ### Initial conditions
# config['ICParameters'] = {'n': 10, 'R': 0.1, 'r': 0.003}    ### Optional parameters to IC
# config['ICParameters'] = {'n': 3, 'eps': config['eps']}    ### Optional parameters to IC
### Working connfigureation sets:

## 3 Bubbles:
# config['ICParameters'] = {'n': 3, 'seed': 0}    ### Optional parameters to IC
# config['ICParameters'] = {'n': 3, 'seed': 1}    ### Optional parameters to IC
# config['ICParameters'] = {'n': 3, 'seed': 42}    ### Optional parameters to IC

## 10 Bubbles:
#config['ICParameters'] = {'n': 10, 'eps': 0.04, 'R': 0.2, 'r': 0.005, 'seed': 3}    ### Optional parameters to IC
#config['ICParameters'] = {'n': 10, 'eps': 0.04, 'R': 0.1, 'r': 0.08, 'seed': 4}    ### Optional parameters to IC

## 30 Bubbles
# config['ICParameters'] = {'n': 30, 'eps': 0.02, 'R': 0.05, 'r': 0.02, 'seed': 0}    ### Optional parameters to IC

## 80 Bubbles
# config['ICParameters'] = {'n': 80, 'eps': 0.01, 'R': 0.03, 'r': 0.01, 'seed': 0}    ### Optional parameters to IC

## 400 Bubbles
#config['ICParameters'] = {'n': 400, 'eps': 0.01, 'R': 0.002, 'r': 0.001, 'seed': 0}    ### Optional parameters to IC

# config['ICParameters'] = {'n': 10, 'R': 0.1, 'r': 0.003}    ### Optional parameters to IC

config['ExportFolder']= os.path.join('./CahnHilliard', config['IC'])
# Generate meaningfull file prefix
# if config['ExportFolder']:
    # Save config to file
    # Force config prefix 
    # config['FilePrefix']   = 'CH_a_{alpha:.3f}_T_{FinalTime:f}_nt_{nTimeSteps:d}_Nx_{Nx:d}_e_{eps:.4f}'.format(**config)

# import pdb; pdb.set_trace()
pb = CahnHilliardR1(**config)

# pb.TS.RA.plot() ### verify the rational approximation

####################################
#       Computational body
####################################

Solution = pb()  ### Numerical solution: generator, to get a list, call [pb()]

t  = np.zeros(pb.TS.nTimeSteps+1) # time
m  = np.zeros(pb.TS.nTimeSteps+1) # mass
Rh = np.zeros(pb.TS.nTimeSteps+1) # Roughness
Rs = np.zeros(pb.TS.nTimeSteps+1) # Roughness
E0 = np.zeros(pb.TS.nTimeSteps+1) # energy (classical)
Em = np.zeros(pb.TS.nTimeSteps+1) # modified energy (see the paper)
Eh = np.zeros(pb.TS.nTimeSteps+1) # history energy (see the paper)
 
if prefix := config['ExportFolder']:
    # Save initial value
    pb.Exporter.export_iv_vtk()

try:
    ### Time integration
    for i, u in tqdm(enumerate(Solution), total=pb.TS.nTimeSteps+1):
        # import pdb; pdb.set_trace()    
        t[i]  = pb.TS.CurrTime
        m[i]  = pb.mass
        Rh[i] = pb.roughness
        # Rs[i] = pb.Residual
        E0[i] = pb.Energy
        # Em[i] = pb.EnergyModified
        # Eh[i] = pb.HistoryEnergy
        print(f"Time step {i}: {t[i]}")
        print(f"   mass: {m[i]}, Energy: {E0[i]}, Roughness: {Rh[i]}")
        if np.isnan([m[i],Rh[i],E0[i]]).any():
            raise Exception('Some of the characteristic quantities are NaN. Consider decreasing the time-step size.')
                           ### YOUR CODE: do something with u (solution approximation at t_i)
        pb.Exporter.export_step_vtk(**config)

finally:
    ### Save calculated values to file 
    # t = np.linspace(pb.TS.InitTime,pb.TS.FinalTime, pb.TS.nTimeSteps+1)
    np.savetxt(os.path.join(config['ExportFolder'], pb.Exporter.FilePrefix + '_data.txt'), np.transpose([t, m, E0, Rh, Em]), header='t,Mass,Energy,Roughness,Modified Energy',comments='', delimiter=',', fmt='%s')
            
    ####################################
    #       Plots
    ####################################


    plt.figure("Mass evolution")
    plt.plot(t,m)
    plt.title("Mass evolution")
    plt.xlabel("time")
    plt.ylabel("mass")
    plt.grid(True)
    plt.savefig(os.path.join(config['ExportFolder'], pb.Exporter.FilePrefix + '_Mass.png'))
    # plt.show()

    plt.figure("Energy evolution")
    plt.plot(t,E0)
    # plt.title("Time evolution")
    plt.xlabel("time")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.savefig(os.path.join(config['ExportFolder'], pb.Exporter.FilePrefix + '_Energy.png'))

    #NOTE! Residual is not working properly yet
    # plt.figure("Residual of stationary part")
    # plt.plot(t,Rs)
    # plt.title("Residual of stationary part")
    # plt.xlabel("time")
    # plt.ylabel("Residula")
    # plt.grid(True)
    # plt.savefig(os.path.join(config['ExportFolder'], pb.Exporter.FilePrefix + '_Residual.png'))
    
    # plt.figure("Roughness of solution over time")
    # plt.plot(t,Rh)
    # # plt.title("Time evolution")
    # plt.xlabel("time")
    # plt.ylabel("Roughness")
    # plt.grid(True)
    # plt.savefig(os.path.join(config['ExportFolder'], pb.Exporter.FilePrefix + '_Roughness.png'))

    # plt.figure("Modified Energy evolution")
    # plt.subplot(121)
    # plt.plot(t,E0, "-", label="Energy (std)")
    # plt.plot(t,Em, "--", label="Energy (modified)")
    # plt.legend()
    # plt.title("Energy evolution")
    # plt.xlabel("time")
    # plt.ylabel("energy")

    # plt.subplot(122)
    # plt.plot(t,Eh, "-", label="History energy")
    # plt.legend()
    # plt.title("History energy")
    # plt.xlabel("time")
    # plt.ylabel("energy")
    # plt.savefig(os.path.join(config['ExportFolder'], pb.Exporter.FilePrefix  + '_Energy_compound.png'))
    # plt.grid(True)
    plt.show()

    # import pdb; pdb.set_trace()    

