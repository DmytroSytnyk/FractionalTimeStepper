
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/ustim/Projects/CODES/FDE/fde_code_github")
from source.TimeStepper.RationalApproximation import RationalApproximation_AAA as RationalApproximation

EXPORT_PATH = '/home/ustim/Projects/FDE/fde_ra/IMAJNA/data/'

T       = 1
h       = 1.e-5
AAA_tol = 1.e-12

config = {
    'FinalTime'     :   T,
    'nSupportPoints':   100,
    'tol'           :   AAA_tol,
    'verbose'       :   False,
}


ls_alpha = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
w_min, w_max, lmbda_min, lmbda_max = [], [], [], []


for a in ls_alpha:
    config['alpha'] = a
    RA = RationalApproximation(**config, Zmin=1/T, Zmax=1/h)
    w, lmbda, w_inf = RA.c, RA.d, RA.c_inf
    w_min.append( w.min() )
    w_max.append( w.max() )
    lmbda_min.append( lmbda.min() )
    lmbda_max.append( lmbda.max() )

output = np.array([ls_alpha, w_min, w_max, lmbda_min, lmbda_max]).T
np.savetxt(EXPORT_PATH + 'data_weights_poles_bounds', output)

plt.figure()
plt.plot(ls_alpha, w_min, 'o-', label='min weight')
plt.plot(ls_alpha, w_max, 'o-', label='max weight')
plt.legend()
plt.yscale('log')
plt.ylabel(r'$w_k$')
plt.xlabel(r'$\alpha$')

plt.figure()
plt.plot(ls_alpha, lmbda_min, 'o-', label='min exponent')
plt.plot(ls_alpha, lmbda_max, 'o-', label='max exponent')
plt.legend()
plt.yscale('log')
plt.ylabel(r'$\lambda_k$')
plt.xlabel(r'$\alpha$')

plt.show()