
from math import *
import numpy as np
import dolfin as dl
import scipy
from scipy import optimize
from scipy.special import binom, gamma
from time import time
import matplotlib.pyplot as plt
from sympy import denom
from torch import numel

from .BasicTS import BasicTS
from .RationalApproximation import RationalApproximation_AAA as RationalApproximation
from .SingularityCorrection import CorrectionTerm




#############################################################################################
#
#   Fractional Time-Stepper class using Rational Approximation
#
#############################################################################################

class FracTS_Blended(BasicTS):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        alpha = kwargs.get("alpha", 1)
        self.set_alpha(alpha)
        self.nModes = kwargs.get("nModes", 20)
        self.blending_coefficient = kwargs.get("blending_coefficient", 1)
        self.compute_Coefficients(**kwargs)

        self.fg_correction = kwargs.get("correction", False)

    #----------------------------------------------------------------------------------------
    #   Compute the qadrature nodes and weights
    #----------------------------------------------------------------------------------------
    def compute_Quadrature(self, **kwargs):        
        if "nModes" in kwargs.keys():
            self.nModes = kwargs["nModes"]

        self.quad = kwargs.get('quad', 'GJ') ### 'GJ', 'GJL'

        if self.quad == 'GJ':
            if self.alpha == 1:
                theta_k = np.array([0.])
                w_k = np.array([1.])
            else:
                ### Gauss-Jacobi quadrature
                x_k, w_k = scipy.special.roots_jacobi(self.nModes, self.alpha-1, -self.alpha)
                theta_k = (x_k+1)/2
                w_k = w_k * np.sin(pi*self.alpha)/pi

        elif self.quad == 'GJL':
            ### Gauss-Jacobi-Lobatto quadrature
            assert(self.nModes > 1)
            n, a, b = self.nModes-1, self.alpha-1, -self.alpha
            x_k, w_k = scipy.special.roots_jacobi(n, 1+a, 1+b)
            theta_k = np.zeros(self.nModes+1)
            w_k     = np.zeros(self.nModes+1)
            # w1_k    = np.zeros(self.nModes+1)
            theta_k[0], theta_k[-1] = 0, 1
            theta_k[1:-1] = (x_k+1)/2
            w_k[0]  = self.alpha * binom(n+a+1, n)/binom(n+b+1, n)/binom(n+1, n)
            w_k[-1] = (1-self.alpha) * binom(n+b+1, n)/binom(n+a+1, n)/binom(n+1, n)
            # w1_k[0], w1_k[-1]  = w_k[0], w_k[-1]
            for k in range(n):
                if a>-1 and b>-1:
                    Pnk = scipy.special.eval_jacobi(n+1, a, b, x_k[k])
                else:
                    Pnk = 1
                # aux_coef = (1-x_k[k]**2) / ( (4*(n+a+1)*(n+b+1) + (a-b)**2)/(2*n+1)**2 - x_k[k]**2 )
                # w1_k[k+1] = self.alpha*(1-self.alpha)/(n+1)**2 * binom(n+a+1, n)*binom(n+b+1, n)/binom(n+1, n) * aux_coef / Pnk**2
                w_k[k+1] = np.sin(pi*self.alpha)/pi * gamma(n+a+2)*gamma(n+b+2) / factorial(n)**2 / (n+1)**3 / Pnk**2

        if self.verbose:
            print('Number of modes = ', self.nModes)
            print('Nodes  : ', theta_k)
            print('Weights: ', w_k)
        
        return theta_k, w_k

    #----------------------------------------------------------------------------------------
    #   Compute coefficients
    #----------------------------------------------------------------------------------------
    def compute_Coefficients(self, **kwargs):
        if "blending_coefficient" in kwargs.keys():
            self.blending_coefficient = kwargs["blending_coefficient"]
        self.theta_k, self.w_k = self.compute_Quadrature(**kwargs)
        self.nModes = len(self.theta_k)

        ### Aliases
        th_k, gm, h = self.theta_k, self.blending_coefficient, self.dt
        
        # alpha = self.alpha
        # lmbda_k = th_k / (1-th_k)
        # gm = th_k #h**(alpha-1) * (1 - np.exp(-lmbda_k))

        ### Coefficients
        denominator = (1-th_k)*((1-th_k) + th_k*gm*h) + th_k*(0.5*h)*((1-th_k) + 2*th_k*gm*h)
        self.a_k = 0.5*h*((1-th_k) + 2*th_k*gm*h) / denominator
        self.b_k = (1-th_k)*(((1-th_k) + th_k*gm*h) - th_k*(0.5*h)) / denominator
        self.c_k = 0.5*h*(1-th_k) / denominator

        self.a_bar = np.sum(self.w_k * self.a_k)
        self.c_bar = np.sum(self.w_k * self.c_k)
        

        self.Scheme = kwargs.get('scheme', 'GJ:mCN')

    #----------------------------------------------------------------------------------------
    #   Updates
    #----------------------------------------------------------------------------------------   

    def update_Modes(self):
        self.Modes[:] = self.b_k*self.Modes - (self.a_k*self.CurrF[:,None] + self.c_k*self.PrevF[:,None])
        if self.fg_correction:
            self.Modes[:] += (self.a_k+self.c_k) * self.CorrectionTerm.CorrectorModes()

    def update_History(self):
        self.History[:] = self.Modes @ (self.w_k * self.b_k)

    def update_Tangent(self):
        if self.Model.MatrixUpdate:
            M = self.Model.MassMatrix
            K = self.Model.StiffMatrix
            A = M + self.a_bar*K
        else:
            A = None
        return A

    def update_Residual(self):
        if hasattr(self.Model, "source"):
            t = self.CurrTime
            f = self.Model.source(t)
        else:
            f = 0
        if self.fg_correction:
            f -= self.CorrectionTerm.CorrectorF()
        # Res = self.Mu0 + self.a_bar*f - self.c_bar*self.PrevF + self.History
        Mu  = self.Model.multMass(self.CurrSol)
        Res = Mu - self.Mu0 + self.a_bar*self.CurrF + self.c_bar*self.PrevF - self.History

        if self.fg_correction:
            Res -= self.CorrectionTerm.CorrectorResidual()
        return Res

    def update_NonLinearPart(self):
        self.CurrF[:] = self.Model.update_NonLinearPart(self.CurrSol, time=self.CurrTime)
        if self.fg_correction:
            self.CurrF[:] += self.CorrectionTerm.CorrectorF()


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
        self.PrevF3  = np.zeros_like(u)
        self.CurrFH  = np.zeros_like(u)
        self.PrevFH  = np.zeros_like(u)
        self.Residual= np.zeros_like(u)
        self.History = np.zeros_like(u)
        self.Modes   = np.zeros([u.shape[0], self.nModes])


        ### Initial data
        self.restart()
        self.InitSol[:] = kwargs.get("u_init", self.Model.ArrInitSol) 
        self.CurrSol[:] = self.InitSol[:]
        self.PrevSol[:] = self.InitSol[:]
        self.Mu0        = self.Model.multMass(self.InitSol)

        if self.fg_correction:
            self.CorrectionTerm = CorrectionTerm(self)


        self.update_NonLinearPart()
        self.PrevF[:] = self.CurrF[:]

        if self.fg_correction and not self.ReferenceSolution:
            self.CorrectionTerm.init_Modes()

        self.update_History()
        
        if hasattr(self.Model, 'MatrixUpdate'):
            self.Model.MatrixUpdate = True

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

        ReferenceSolution = kwargs.get("ReferenceSolution", False)
        self.ReferenceSolution = ReferenceSolution

        self.init_computation(**kwargs)
        yield self.OutputFormat(self.InitSol)

        ### Time-Stepping
        while self.inc():

            if ReferenceSolution:
                self.CurrSol[:] = self.Model.ReferenceSolution(self.CurrSol[:], t=self.CurrTime)
            else:

                ### Non-Linear solver
                self.CurrSol[:] = self.NLstep()
                # self.NLIter = 0
                # self.CurrSol[:] = self.NLSolve(self.LinearSolve, self.CurrSol[:])

                if self.verbose:
                    print('Time step {0:d}, iter {1:d}'.format(self.TimeStep, self.NLIter))

                self.update_Fields()

                self.update_QoIs()

                self.UserMonitor(**kwargs)

            yield self.OutputFormat(self.CurrSol)

    def NLstep(self):
        self.NLIter = 0
        self.CurrSol[:] = self.NLSolve(self.LinearSolve, self.CurrSol[:])
        return self.CurrSol[:]


    #----------------------------------------------------------------------------------------
    #   Linear solver functionfor the Fixed-Point iteration
    #----------------------------------------------------------------------------------------
    def LinearSolve(self, y):
        self.update_NonLinearPart()
        # if self.fg_correction:
        #     self.CorrectionTerm.update()
        TangentMatrix    = self.update_Tangent()
        self.Residual[:] = self.update_Residual()
        # self.CurrSol[:]  = self.LSolve(TangentMatrix, self.CurrSol, self.Residual)
        x0 = 1*self.CurrSol    
        self.CurrSol[:] -= self.LSolve(TangentMatrix, x0, self.Residual)
        self.NLIter += 1
        return self.CurrSol[:]


    #----------------------------------------------------------------------------------------
    #   Update the fields for the current time-step
    #----------------------------------------------------------------------------------------
    def update_Fields(self):
        self.update_NonLinearPart()
        self.update_Modes()
        self.update_History()
        self.PrevSol[:] = self.CurrSol[:]
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
            #NOTE: User's code here
            pass

    

#############################################################################################









