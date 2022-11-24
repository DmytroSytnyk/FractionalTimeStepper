# import random
import numpy as np
import random
from dolfin import *
from tqdm import tqdm
from copy import copy
from scipy.special import gamma
import sys, os

from source.TimeStepper import FracTS_RA, FracTS_L1


###########################################
# Change log levele for FEniCS

import logging
ffc_logger = logging.getLogger('FFC')
print('Change FFC log level to Warning')
print('Log level was:', logging.getLevelName(ffc_logger.getEffectiveLevel()))
ffc_logger.setLevel(logging.WARNING)
print('Log level is now:', logging.getLevelName(ffc_logger.getEffectiveLevel()))
###########################################

###########################################
# Form compiler options
# are supposed to be defined in the caller script

parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

###########################################

##########################################

from .CahnHilliardR1 import CahnHilliardR1 
from .CahnHilliardR2 import CahnHilliardR2 
from .CahnHilliardR4 import CahnHilliardR4 
from .CahnHilliardR5 import CahnHilliardR5 





#######################################################################################################
#	Periodic boundary conditions
#######################################################################################################

class PeriodicBoundary(SubDomain):

	def __init__(self, ndim):
		SubDomain.__init__(self)
		self.ndim = ndim

	def inside(self, x, on_boundary):
		return bool( 	any([ near(x[j], 0) for j in range(self.ndim) ]) and 
					not any([ near(x[j], 1) for j in range(self.ndim) ]) and 
						on_boundary
					)

	def map(self, x, y):
		for j in range(self.ndim):
			if near(x[j], 1):
				y[j] = x[j] - 1
			else:
				y[j] = x[j]






###########################################

