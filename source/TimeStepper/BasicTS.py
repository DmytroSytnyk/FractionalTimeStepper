
from math import *
import numpy as np
import dolfin as dl
from scipy import optimize
from time import time
import matplotlib.pyplot as plt

#############################################################################################
#
#   Basic Fractional Time-Stepper class (abstract)
#
#############################################################################################

class BasicTS:

    def __init__(self, **kwargs):
        self.verbose = kwargs.get("verbose", False)

        self.InitTime  = kwargs.get('InitTime',  0)
        self.FinalTime = kwargs.get('FinalTime', 1)
        assert(self.InitTime < self.FinalTime)
        self.TimeInterval = self.FinalTime - self.InitTime
        self.nTimeSteps   = kwargs.get('nTimeSteps',  100)
        self.set_TimeSolver(self.nTimeSteps)
        self.theta = kwargs.get('theta',  1)

        self.restart()

        self.Model = kwargs.get('Model', None)
        if self.Model is not None:
            self.domain_shape = self.Model.shape
            self.ndim = len(self.domain_shape)
            self.ndofs_space = np.prod(self.domain_shape)

            self.ndofs_local = self.Model.ArrSol.size
            # self.ndofs_total = self.ndofs_local
            # self.dofs = [ np.arange(self.ndofs_local[i]) + np.sum(self.ndofs_local[:i]) for i in range(self.Neq) ]

            lspace = [ np.linspace(0,1,N) for N in self.domain_shape ]
            self.grid_space = np.meshgrid(*lspace, indexing='xy')

            ### Solvers
            self.NLSolve = self.Model.NLSolve or optimize.fixed_point
            self.LSolve  = self.Model.LSolve or self.set_linSolver()        
            self.MatrixUpdate = self.Model.MatrixUpdate or False if hasattr(self.Model, 'MatrixUpdate') else False

    def set_alpha(self, alpha):
        assert(alpha>=0 and alpha<=1)
        self.alpha = alpha
        self.Model.alpha = alpha

    #----------------------------------------------------------------------------------------
    #   Set Time-Stepper options
    #----------------------------------------------------------------------------------------
    def set_TimeSolver(self, nTimeSteps=None):
        if nTimeSteps:
            self.nTimeSteps = nTimeSteps
            self.dt = self.TimeInterval / self.nTimeSteps
            self.MaxTimeStep = self.nTimeSteps

    #----------------------------------------------------------------------------------------
    #   Restart
    #----------------------------------------------------------------------------------------
    def restart(self):
        self.TimeStep = 0
        self.CurrTime = self.InitTime
        self.PrevTime = self.InitTime

    #----------------------------------------------------------------------------------------
    #   Increment
    #----------------------------------------------------------------------------------------
    def inc(self):
        self.TimeStep += 1
        self.PrevTime = self.CurrTime  
        self.CurrTime = self.InitTime + self.TimeStep*self.dt
        return not self.stop()

    #----------------------------------------------------------------------------------------
    #   Stop criterion
    #----------------------------------------------------------------------------------------
    def stop(self):
        return self.TimeStep > self.MaxTimeStep
        # return self.CurrTime > self.FinalTime 

    
    #----------------------------------------------------------------------------------------
    #   Set default linear solver
    #----------------------------------------------------------------------------------------
    def set_linSolver(self):
        # solver = dl.PETScLUSolver("mumps")
        # solver = dl.PETScKrylovSolver("gmres", "amg")
        solver = dl.PETScKrylovSolver("cg", "amg")
        solver.parameters["maximum_iterations"] = 1000
        solver.parameters["relative_tolerance"] = 1.e-6
        solver.parameters["absolute_tolerance"] = 1.e-8
        solver.parameters["error_on_nonconvergence"] = True
        solver.parameters["nonzero_initial_guess"] = True
        solver.parameters["monitor_convergence"] = False
        return solver


    #----------------------------------------------------------------------------------------
    #   Output the solution in the geometrically consistent numpy array format
    #   (finaloutput format)
    #----------------------------------------------------------------------------------------
    def OutputFormat(self, y):
        if hasattr(self.Model, 'v2d'):
            v2d, shp = self.Model.v2d, self.domain_shape
            # # return [ y[i][v2d].reshape(shp) for i in range(self.Neq) ]
            # return [ y[i][v2d] for i in range(self.Neq) ]
            return y[v2d].reshape(shp)
        else:
            return 1*y

#############################################################################################