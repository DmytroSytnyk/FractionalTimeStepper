
from math import *
import numpy as np
import dolfin as dl
from scipy.special import gamma
import matplotlib.pyplot as plt
from time import time

import sys
sys.path.append("/home/khristen/Projects/FDE/code/source/")

from Models.DiffusionModel1D import DiffusionModel1D
from Models.SimpleScalarModel import SimpleScalarModel
from TimeStepper import FracTS_RA, FracTS_L1
from MittagLeffler import ml

####################################

def g(x, alpha):
    y = x**(alpha-1) / gamma(alpha)
    return y

####################################
# Fractional kernels
####################################


ls_alpha = [0.1, 0.3, 0.5, 0.7, 0.9, 1]

x = np.linspace(0.05, 1, 100)

for alpha in ls_alpha:
    plt.plot(x, g(x, alpha), label=r'$\alpha={0}$'.format(alpha))

plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$g_{\alpha}(t)$')
plt.show()


####################################
# Gamma function
####################################

# x = np.linspace(0.01, 1, 100)
# plt.plot(x, gamma(x), 'r-')
# plt.ylim([0,100])
# plt.xlabel(r'$x$')
# plt.ylabel(r'$\Gamma(x)$')
# plt.show()


