
from math import *
import numpy as np
import dolfin as dl
from scipy import optimize
from time import time
import matplotlib.pyplot as plt

from .BasicTS import BasicTS
from .RationalApproximation import RationalApproximation_AAA as RationalApproximation


#############################################################################################
#
#   Fractional Time-Stepper class using L1 scheme
#
#############################################################################################

class FracTS_L1(BasicTS):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.alpha = kwargs.get("alpha", 1)
        assert(self.alpha>=0 and self.alpha<=1)

        self.Scheme = 'L1'

        self.compute_Coefficients(**kwargs)
        self.compute_AdamsMoultonCoefficients()
        if self.verbose:
            print()
            print('=================================')
            print('Fractional Derivative scheme: L1')
            print('=================================')
            print('alpha:             ', self.alpha)
            print('Number time-steps: ', self.nTimeSteps)
            print()


    #----------------------------------------------------------------------------------------
    #   Compute coefficients
    #----------------------------------------------------------------------------------------
    def compute_Coefficients(self, **kwargs):
        alpha, h = self.alpha, self.dt
        th = 1

        b0 = 1/gamma(2-alpha)
        self.b = np.array([ b0 ])

        self.beta1 = h**alpha / b0 * th
        self.beta2 = h**alpha / b0 * (1-th)


    def compute_AdamsMoultonCoefficients(self):
        alpha, h = self.alpha, self.dt

        self.AM1 = h**alpha/gamma(1+alpha)

        self.AM2 = [0]*2
        self.AM2[0] = h**alpha / gamma(2+alpha)
        self.AM2[1] = alpha*(h**alpha) / gamma(2+alpha)

        self.AM3 = [0]*3
        self.AM3[0] = h**alpha * ( 1/(2*gamma(2+alpha)) + 1/gamma(3+alpha) )
        self.AM3[1] = h**alpha * ( 1/gamma(1+alpha) - 2/gamma(3+alpha) )
        self.AM3[2] = h**alpha * ( -1/(2*gamma(2+alpha)) + 1/gamma(3+alpha) )




    #----------------------------------------------------------------------------------------
    #   Initialize the time integration
    #----------------------------------------------------------------------------------------
    def init_computation(self, **kwargs):

        ### Initialize vectors
        u = self.Model.ArrSol
        self.InitSol = np.zeros_like(u)
        self.CurrSol = np.zeros_like(u)
        self.PrevSol = np.zeros_like(u)
        self.CurrF   = np.zeros_like(u)
        self.PrevF   = np.zeros_like(u)
        self.Residual= np.zeros_like(u)
        self.History = np.zeros_like(u)
        self.HistSol = np.zeros([u.size,0])


        ### Initial data
        self.restart()
        self.InitSol[:] = kwargs.get("u_init", self.Model.ArrInitSol) 
        self.CurrSol[:] = self.InitSol[:]
        self.PrevSol[:] = self.InitSol[:]
        self.CurrF[:]   = self.Model.update_NonLinearPart(self.CurrSol, time=self.CurrTime)
        self.PrevF[:]   = self.CurrF[:]
        self.HistSol    = np.hstack([ self.HistSol, self.Model.multMass(self.CurrSol).reshape([-1,1]) ])
        self.History[:] = self.HistSol[:,0]

        self.b = np.array([self.b[0]])
        # alpha  = self.alpha
        # j  = len(self.b)
        # bj = ((j+1)**(1-alpha) - j**(1-alpha)) / gamma(2-alpha)
        # self.b = np.append(self.b, bj)
        # coefs = np.append(-np.diff(self.b)/self.b[0], self.b[-1]/self.b[0])[::-1]
        
        if hasattr(self.Model, 'MatrixUpdate'): self.Model.MatrixUpdate = True

        ### Initialize QoIs
        self.QoIs = [None] * (self.MaxTimeStep+1)
        self.update_QoIs()


    #----------------------------------------------------------------------------------------
    #   Run the time integration
    #----------------------------------------------------------------------------------------
    def __call__(self, **kwargs):
        nTimeSteps = kwargs.get("nTimeSteps", self.nTimeSteps)
        self.set_TimeSolver(nTimeSteps)

        self.MaxTimeStep = kwargs.get("MaxTimeStep", self.nTimeSteps)

        self.init_computation(**kwargs)
        yield self.OutputFormat(self.InitSol)

        ReferenceSolution = kwargs.get("ReferenceSolution", False)

        ### Time-Stepping
        while self.inc():

            if ReferenceSolution:
                self.CurrSol[:] = self.Model.ReferenceSolution(self.CurrSol[:], t=self.CurrTime)
            else:

                ### Non-Linear solver
                self.NLIter = 0
                self.CurrSol[:] = self.NLSolve(self.LinearSolve, self.CurrSol[:])
                if self.verbose: print('Time step {0:d}, iter {1:d}'.format(self.TimeStep, self.NLIter))

                self.update_Fields()

                self.update_QoIs()

                self.UserMonitor(**kwargs)

            yield self.OutputFormat(self.CurrSol)


    #----------------------------------------------------------------------------------------
    #   Linear solver functionfor the Fixed-Point iteration
    #----------------------------------------------------------------------------------------
    def LinearSolve(self, y):
        
        if self.Model.MatrixUpdate:
            M = self.Model.MassMatrix
            K = self.Model.StiffMatrix
            A = M + self.beta1*K

            if self.verbose:
                print('betas:', self.beta1, self.beta2)
                print('G1:', self.AM1)
                print('G2:', self.AM2[0], self.AM2[1])
                print('G3:', self.AM3[0], self.AM3[1], self.AM3[2])
        else:
            A = None

        self.Residual[:] = -self.beta2*self.PrevF + self.History
        self.CurrSol[:]  = self.LSolve(A, self.CurrSol, self.Residual)
        
        self.NLIter += 1

        return self.CurrSol[:]


    #----------------------------------------------------------------------------------------
    #   Update the fields for the current time-step
    #----------------------------------------------------------------------------------------
    def update_Fields(self):
        alpha = self.alpha        
        j  = len(self.b)
        bj = ((j+1)**(1-alpha) - j**(1-alpha)) / gamma(2-alpha)
        self.b = np.append(self.b, bj)
        coefs = np.append(-np.diff(self.b)/self.b[0], self.b[-1]/self.b[0])[::-1]

        HistSolNew      = self.Model.multMass(self.CurrSol).copy() 
        self.HistSol    = np.hstack([ self.HistSol,  HistSolNew.reshape([-1,1])])
        self.History[:] = self.HistSol @ coefs

        self.PrevSol[:] = self.CurrSol[:]
        self.CurrF[:]   = self.Model.update_NonLinearPart(self.CurrSol, time=self.CurrTime)
        self.PrevF[:]   = self.CurrF[:]


    #----------------------------------------------------------------------------------------
    #   Update the quanities of interest
    #----------------------------------------------------------------------------------------
    def update_QoIs(self):
        if hasattr(self.Model, 'update_QoIs'):
            self.QoIs[self.TimeStep] = self.Model.update_QoIs(self.CurrSol)
        return self.QoIs


    #----------------------------------------------------------------------------------------
    #   The routine containing all the user-defined monitoring functions and options
    #----------------------------------------------------------------------------------------
    def UserMonitor(self, **kwargs):
        if kwargs.get("my_flag"):
            # CODE HERE
            pass


#############################################################################################


