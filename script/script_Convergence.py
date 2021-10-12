
from math import *
import numpy as np
import dolfin as dl
from scipy.special import gamma
import matplotlib.pyplot as plt
from time import time

import sys
sys.path.append("/home/khristen/Projects/FDE/FDE_RA/code/source/current")

from Models.DiffusionModel1D import DiffusionModel1D
from Models.SimpleScalarModel import SimpleScalarModel
from TimeStepper import FracTS_RA, FracTS_L1
from MittagLeffler import ml

####################################
#           Parameters
####################################

model_type = 'scalar' #'diffusion1d' # 'scalar', 'diffusion1d'

def src(t,x):
    return np.exp(alpha*t)

config = {
    'FinalTime'     :   1,
    'stiff'         :   1,
    'BC'            :   ['Dirichlet', '0'],
    'source'        :   '0',
    'shape'         :   [1000], ### domain shape
    'nSupportPoints':   100,
    # 'scheme'  :  'mCN', # 'mIE' - modified Implicit Euler (theta=1), 'mCN' - modified Crank-Nicolson, 'L1'
    'tol'           :   1.e-13,
    'verbose'       :   False,
}

if model_type == 'scalar':
    config['IC'] = 1
    mdl = SimpleScalarModel(**config)
elif model_type == 'diffusion1d':
    config['IC'] = 'sin(pi*x[0])'
    mdl = DiffusionModel1D(**config)
else:
    raise Exception('Unknown model!')



# IMPORT_PATH = '/home/khristen/Projects/FDE/FDE_RA/code/data/Heat/'
# EXPORT_PATH = '/home/khristen/Projects/FDE/FDE_RA/draft_siam/data/'
fg_EXPORT = False


# ls_ALPHAS  = [1, 0.8, 0.5, 0.3, 0.1, 0.03, 0.01]
ls_ALPHAS  = [0.5]
ls_SCHEMES = ['L1']
ls_NITER   = [100*int(2**l) for l in range(0,6)]
# ls_NITER   = [1000]

verbose = False
doPlots = False
doPlots_err = False


####################################

u_init = mdl.ArrInitSol

def norm(u):
    return np.linalg.norm(u, ord=2)

def normH(u):
    # return np.linalg.norm(u, ord=2)
    return mdl.norm(u)

def normT(u):
    # return np.linalg.norm(u, ord=2)
    # return np.linalg.norm(u, ord=np.Inf)
    n = len(u)//2
    return np.linalg.norm(u[n:], ord=2)
    # return np.abs(u[-1])


####################################
#       Computational body
####################################


for scheme in ls_SCHEMES:
    config['scheme'] = scheme

    for alpha in ls_ALPHAS:
        config['alpha'] = alpha
        mdl.alpha = alpha

        ls_dt  = []
        ls_err = []

        for nIter in ls_NITER:
            config['nTimeSteps'] = nIter


            if config['scheme'] == 'L1': 
                pb = FracTS_L1(**config, Model=mdl)
            else:
                pb = FracTS_RA(**config, Model=mdl)


            print('-------------------------')
            print('scheme : {0}'.format(pb.Scheme))
            print('alpha  : {0}'.format(pb.alpha))
            print('dt     : {0}'.format(pb.dt))


            ExactSolution = list(pb(ReferenceSolution=True))
            AppxSolution  = pb()
            norm_ref = np.zeros(nIter+1)
            norm_RA  = np.zeros(nIter+1)
            err_RA   = np.zeros(nIter+1)

            # if doPlots:
            #     fig = plt.figure('Solution ({0} scheme)'.format(scheme))
            #     plt.ion()
            # else:
            #     plt.ioff()

            ### Time integration
            for i, u in enumerate(AppxSolution):
                u_ref = ExactSolution[i]
                norm_ref[i] = normH(u_ref-u_init)
                norm_RA[i]  = normH(u-u_init)
                err_RA[i]   = normH(u-u_ref)

                if doPlots:
                    fig = plt.figure('Evolution')
                    # plt.cla()
                    plt.ylim((-0.01, pb.InitSol.max() + 0.01))
                    plt.grid()
                    x = pb.grid_space[0]
                    plt.plot(x, u, '-')
                    # plt.plot(x, u_ref, 'r--')
                    plt.xlabel('x')
                    plt.ylabel('u')
                    # plt.legend(['RA', 'Ref'])
                    # plt.title('Time: {0:f} s'.format(pb.CurrTime))
                    fig.canvas.draw()
                    plt.pause(0.01)
                    plt.waitforbuttonpress()
            
            ls_dt  = np.append(ls_dt, pb.dt)
            ls_err = np.append(ls_err, normT(err_RA)/normT(norm_ref))

            ###-----------------------

            if doPlots_err:
                t = np.arange(nIter+1)*pb.dt
                plt.figure('Error, dt={0:.2e}'.format(pb.dt))
                plt.subplot(1,2,1)
                plt.plot(t[1:], err_RA[1:]/norm_ref[1:],'b-')
                plt.legend(['RA'])
                plt.xlabel('time, [s]')
                plt.ylabel('relative error, (l2 in space)')
                plt.yscale('log')
                plt.subplot(1,2,2)
                plt.plot(t[1:], norm_RA[1:],'b-')
                plt.plot(t[1:], norm_ref[1:],'r--')
                plt.legend(['RA','Ref'])
                plt.xlabel('time, [s]')
                plt.ylabel('Solution norm, (l2 in space)')
                # plt.yscale('log')
                plt.ioff()
                plt.show()

        if scheme == 'mIE':
            ls_slope = np.array([h for h in ls_dt])
            label_slope = r'$h^1$'
        elif scheme == 'mCN':
            ls_slope = np.array([h**(1+alpha) for h in ls_dt])
            label_slope = r'$h^{1+\alpha}$'
        elif scheme == 'L1':
            ls_slope = np.array([h**(1) for h in ls_dt])
            label_slope = r'$h^{2-\alpha}$'
        else:
            raise Exception('Unknown scheme')
        ls_slope = (ls_err[0]/ls_slope[0]) * ls_slope

        ### Export to the draft
        if fg_EXPORT:
            output = np.array([ls_dt[::-1], ls_err[::-1], ls_slope[::-1]]).T
            np.savetxt(EXPORT_PATH + 'data_Heat_Convergence_alpha_{0:d}_{1}'.format(int(100*alpha), scheme), output)


        plt.figure('Convergence ({0} scheme)'.format(scheme))
        if config['scheme'] == 'L1':
            plt.plot(ls_dt, ls_err, 'o-', label=r'L1 $\alpha=${0}'.format(alpha))
        else:
            plt.plot(ls_dt, ls_err, 'o-', label=r'RA $\alpha=${0}'.format(alpha))
        # plt.plot(ls_dt, [ls_err[0]/ls_dt[0]**1*dt for dt in ls_dt], 'k--', label=r'$h$')
        # plt.plot(ls_dt, [ls_err[0]/ls_dt[0]**2*dt**2 for dt in ls_dt], 'k-', label=r'$h^2$')
        plt.plot(ls_dt, ls_slope, '--', label=label_slope)
        plt.legend()
        # plt.title(r'$\alpha={0}$'.format(alpha))
        plt.grid()
        plt.xlabel('dt, [s]')
        plt.ylabel('relative error in T, (l2 in space)')
        plt.yscale('log')
        plt.xscale('log')

        print('\n\n----------------------------------------------')
        print('Convergence rate: ', np.log2(ls_err[:-1]/ls_err[1:]))
        print('Expected rate: ', 1+alpha)

plt.show()


