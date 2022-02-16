
from math import *
import numpy as np
import dolfin as dl
from scipy.special import gamma
import matplotlib.pyplot as plt
from matplotlib.ticker import (LinearLocator, MultipleLocator, FormatStrFormatter, AutoMinorLocator, AutoLocator)
from time import time
from tqdm import tqdm

import sys

from source.Models.CahnHilliardModel import CahnHilliard, InitialConditions
from source.TimeStepper.FractionalDerivative import FractionalDerivativeRA, FractionalDerivativeL1
from .LoadData import LoadData

####################################
#           Parameters
####################################


config = {
    'alpha'     :   1,
    # 'Method'    :   7,
    'FinalTime' :   2,
    'CH_scheme' :   'Split', #'Stab', #'Secant', #'Split_O2',
    'scheme'    :   'theta', #'RAstab', #'theta', #'Exponential',
    'theta'     :   1,
    'IC'        :   InitialConditions,
    'lmbda'     :   1.e-2,
    'E'         :   1.e2,
    'D'         :   1,
    'S'         :   0,
    'source'    :   '0',
    'Nx'        :   2**6,
    'verbose'   :   False,
}
doPlots_err = True


doRA = True
doL1 = True
doRef = False


niter_list = [(10**k) for k in range(2,3)]



####################################

### Reference solution
if doRef:
    path = './CahnHilliard/Reference_Solution/'
    ref_solution = LoadData(path, ext='npy')
    N_ref = ref_solution.size-1


####################################

### Norms

def norm(u):
    return np.linalg.norm(u, ord=2)

def normH(u):
    return np.linalg.norm(u, ord=2)
    # return np.sum(u)

def normT_weighted(u):
    t = np.arange(len(u))
    return np.linalg.norm(u * t**(1-alpha), ord=2)

def normT(u):
    # return np.linalg.norm(u, ord=2)
    # return np.linalg.norm(u, ord=np.Inf)
    n = len(u)//3
    return np.linalg.norm(u[n:], ord=2)
    # return np.abs(u[-1])
    # return normT_weighted(u)



####################################
#       Computational body
####################################

err_tot = np.array([])
alpha = config['alpha']



for niter in niter_list:
    # problem.set_TS(nTimeSteps=niter)

    config['nTimeSteps'] = niter

    if doL1:
        FracDerL1 = FractionalDerivativeL1(**config)
        problemL1 = CahnHilliard(**config, FD=FracDerL1)
        sol_L1 = problemL1()

    if doRA:
        FracDerRA = FractionalDerivativeRA(**config)
        problemRA = CahnHilliard(**config, FD=FracDerRA)
        sol_RA = problemRA()

    problem = problemRA if doRA else problemL1
    dt = problem.TS.dt
    N  = problem.TS.nTimeSteps
    u_init = problem.c_init

    print('\n-------------------------')
    print('N  = {0:d}'.format(niter))
    print('dt = {0}'.format(dt))
    print('-------------------------')

    if doL1:
        print(FracDerL1.beta1, FracDerL1.beta2)
    if doRA:
        print(FracDerRA.beta1, FracDerRA.beta2)
        print(FracDerRA.G1)
        print(FracDerRA.G2)


    err = np.zeros(niter+1)
    norm_ref = np.zeros_like(err)
    norm_apx = np.zeros_like(err)
    mass_ref = np.zeros_like(err)
    mass_apx = np.zeros_like(err)
    Enrg_ref = np.zeros_like(err)
    Enrg_apx = np.zeros_like(err)



    ### Time integration
    for i in tqdm(range(niter+1), total=niter+1):
        if doRA: u_RA = next(sol_RA)
        if doL1: u_L1 = next(sol_L1)
        u = u_RA if doRA else u_L1

        if doRef:
            i_ref = int(i* N_ref/N)
            u_ref = ref_solution[i_ref]
        else:
            u_ref = u_L1

        Enrg_ref[i] = problem.Energy(u_ref)
        Enrg_apx[i] = problem.Energy(u)
        mass_ref[i] = problem.mass(u_ref)
        mass_apx[i] = problem.mass(u)
        norm_ref[i] = problem.norm(u_ref-u_init) #normH(u_ref-u_init)
        norm_apx[i] = problem.norm(u-u_init) #normH(u-u_init)
        err[i] = problem.norm(u-u_ref) #normH(u-u_ref) #/ norm_ref[i]

    ###-----------------------

    if doPlots_err:
        t = np.arange(niter+1)*dt

        fig, axes = plt.subplots(2, 2) #, sharex=True, sharey=True ) #, figsize=(8, 6), )
        fig.suptitle('Error, dt={0:.2e}'.format(problem.dt), fontsize=15)

        axes[0,0].plot(t[1:], err[1:]/norm_ref[1:],'b-')
        axes[0,0].legend(['RA'])
        axes[0,0].set_xlabel('time, [s]')
        axes[0,0].set_ylabel('relative error, (l2 in space)')
        axes[0,0].set_yscale('log')
        axes[0,0].grid(b=True, which='both', color='gray', linestyle='--')
        axes[0,0].get_xaxis().set_minor_locator(AutoMinorLocator())
        # axes[0,0].get_yaxis().set_minor_locator(AutoMinorLocator())
        # axes[0,0].get_xaxis().set_major_locator(AutoLocator())
        # axes[0,0].get_yaxis().set_major_locator(AutoLocator())

        axes[0,1].plot(t[:], norm_apx[:],'b-')
        axes[0,1].plot(t[:], norm_ref[:],'r--')
        axes[0,1].legend(['RA','Ref'])
        axes[0,1].set_xlabel('time, [s]')
        axes[0,1].set_ylabel('Solution norm, (l2 in space)')
        axes[0,1].grid(b=True, which='both', color='gray', linestyle='--')
        axes[0,0].get_xaxis().set_minor_locator(AutoMinorLocator())
        axes[0,0].get_yaxis().set_minor_locator(AutoMinorLocator())

        axes[1,0].plot(t[:], mass_apx[:],'b-')
        axes[1,0].plot(t[:], mass_ref[:],'r--')
        axes[1,0].legend(['RA','Ref'])
        axes[1,0].set_xlabel('time, [s]')
        axes[1,0].set_ylabel('Mass')
        axes[1,0].grid(b=True, which='both', color='gray', linestyle='--')
        axes[0,0].get_xaxis().set_minor_locator(AutoMinorLocator())
        axes[0,0].get_yaxis().set_minor_locator(AutoMinorLocator())

        axes[1,1].plot(t[:], Enrg_apx[:],'b-')
        axes[1,1].plot(t[:], Enrg_ref[:],'r--')
        axes[1,1].legend(['RA','Ref'])
        axes[1,1].set_xlabel('time, [s]')
        axes[1,1].set_ylabel('Energy')
        axes[1,1].grid(b=True, which='both', color='gray', linestyle='--')
        axes[0,0].get_xaxis().set_minor_locator(AutoMinorLocator())
        axes[0,0].get_yaxis().set_minor_locator(AutoMinorLocator())



        # plt.figure('Error, dt={0:.2e}'.format(problem.dt))

        # plt.subplot(2,2,1)
        # plt.plot(t[1:], err[1:]/norm_ref[1:],'b-')
        # plt.legend(['RA'])
        # plt.xlabel('time, [s]')
        # plt.ylabel('relative error, (l2 in space)')
        # plt.yscale('log')
        

        # plt.subplot(2,2,2)
        # plt.plot(t[:], norm_apx[:],'b-')
        # plt.plot(t[:], norm_ref[:],'r--')
        # plt.gca().grid(b=True, which='minor', color='gray', linestyle='--')
        # plt.legend(['RA','Ref'])
        # plt.xlabel('time, [s]')
        # plt.ylabel('Solution norm, (l2 in space)')

        # plt.subplot(2,2,3)
        # plt.plot(t[:], mass_apx[:],'b-')
        # plt.plot(t[:], mass_ref[:],'r--')
        # plt.grid(b=True, which='minor', color='gray', linestyle='--')
        # plt.legend(['RA','Ref'])
        # plt.xlabel('time, [s]')
        # plt.ylabel('Mass')

        # plt.subplot(2,2,4)
        # plt.plot(t[:], Enrg_apx[:],'b-')
        # plt.plot(t[:], Enrg_ref[:],'r--')
        # plt.grid(b=True, which='minor', color='gray', linestyle='--')
        # plt.legend(['RA','Ref'])
        # plt.xlabel('time, [s]')
        # plt.ylabel('Energy')

        output = np.vstack([t[1:], err[1:]/norm_ref[1:], norm_apx[1:], norm_ref[1:]]).T
        np.savetxt('stability_exp_nu={0:.2f}'.format(alpha), output)

        # plt.show()

    err_tot = np.append(err_tot, normT(err)/normT(norm_ref))


### Convergence plot
T = problem.TS.TimeInterval
dt_list = [T/n for n in niter_list]

plt.figure('Convergence')
plt.plot(dt_list, err_tot, 'bo-')
plt.plot(dt_list, [err_tot[0]/dt_list[0]**1*dt for dt in dt_list], 'k--')
plt.plot(dt_list, [err_tot[0]/dt_list[0]**2*dt**2 for dt in dt_list], 'k-')
# plt.legend(['RA', 'dt', 'dt^2'])
plt.plot(dt_list, [err_tot[0]/dt_list[0]**(1+alpha)*dt**(1+alpha) for dt in dt_list], 'm--')
# plt.plot(dt_list, [dt**(2*nu) for dt in dt_list], 'm-')
# plt.legend(['RA', 'dt', 'dt^2', 'dt^nu', 'dt^(2*nu)'])
plt.legend(['RA', 'dt', 'dt^2', 'dt^nu'])
plt.grid()
plt.xlabel('dt, [s]')
plt.ylabel('relative error in T, (l2 in space)')
plt.yscale('log')
plt.xscale('log')

output = np.vstack([dt_list, err_tot]).T
np.savetxt('CH_conv_nu={0:.2f}_th={1:.1f}'.format(alpha, config['theta']), output)

print('\n\n----------------------------------------------')
print('Convergence rate: ', np.log2(err_tot[:-1]/err_tot[1:]))
print('Expected rate: ', 1+alpha)

plt.show()


