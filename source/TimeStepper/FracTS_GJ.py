
from math import *
import numpy as np
import dolfin as dl
import scipy
from scipy import optimize
from scipy.special import binom, gamma
from time import time
import matplotlib.pyplot as plt

from .BasicTS import BasicTS
from .RationalApproximation import RationalApproximation_AAA as RationalApproximation




#############################################################################################
#
#   Fractional Time-Stepper class using Rational Approximation
#
#############################################################################################

class FracTS_GJ(BasicTS):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.alpha = kwargs.get("alpha", 1)
        assert(self.alpha>=0 and self.alpha<=1)

        self.compute_Coefficients(**kwargs)
        self.compute_AdamsMoultonCoefficients()
        if self.verbose:
            print()
            print('=================================')
            print('Fractional Derivative scheme: GJ')
            print('=================================')
            print('alpha:                ', self.alpha)
            print('Scheme subtype:       ', self.Scheme[3:])
            print('Quadrature type:      ', self.quad)
            print('Number of modes:      ', self.nModes)
            print('Number of time-steps: ', self.nTimeSteps)
            print()


    #----------------------------------------------------------------------------------------
    #   Compute the rational approximation coefficients
    #----------------------------------------------------------------------------------------
    def compute_Quadrature(self, **kwargs):    
        self.nModes = 5    
        self.quad = kwargs.get('quad', 'GJL') ### 'GJ', 'GJL'

        if self.quad == 'GJ':
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

        if 1: #self.verbose:
            print('Number of modes = ', self.nModes)
            print('Nodes  : ', theta_k)
            print('Weights: ', w_k)
        
        return theta_k, w_k

    #----------------------------------------------------------------------------------------
    #   Compute coefficients
    #----------------------------------------------------------------------------------------
    def compute_Coefficients(self, **kwargs):
        theta_k, w_k = self.compute_Quadrature(**kwargs)

        if self.quad == 'GJ':
            c = w_k/(1-theta_k)
            d = theta_k/(1-theta_k)
            h = self.dt
            self.c_inf = 0
        elif self.quad == 'GJL':
            c = w_k[:-1]/(1-theta_k[:-1])
            d = theta_k[:-1]/(1-theta_k[:-1])
            h = self.dt
            self.c_inf = w_k[-1]

        self.Scheme    = kwargs.get('scheme', 'GJ:mCN')
        scheme_subtype = self.Scheme[3:]

        if scheme_subtype == 'mIE':
            th = 1
            self.gamma_k = (1 - d*(1-th)*h) / (1 + d*th*h)
            self.beta1_k  = c*th*h / (1 + d*th*h)
            self.beta2_k  = c*(1-th)*h / (1 + d*th*h)
        elif scheme_subtype == 'mCN':
            self.gamma_k = np.exp(-d*h)
            self.beta1_k = c * np.where(d != 0, (1-(1-np.exp(-d*h))/(d*h))/d, h/2)
            self.beta2_k = c * np.where(d != 0, ((1-np.exp(-d*h))/(d*h)-np.exp(-d*h))/d, h/2)
        elif scheme_subtype == 'CN':
            th = 0.5
            self.gamma_k = (1 - (1-th)*d*h) / (1 + th*d*h)
            self.beta1_k = c * th*h / (1 + th*d*h)
            self.beta2_k = c * (1-th)*h / (1 + th*d*h)
        else:
            raise Exception('Unknown numerical scheme !')

        self.gamma_k = np.insert(self.gamma_k, 0, 1)
        self.beta1_k = np.insert(self.beta1_k, 0, 0)
        self.beta2_k = np.insert(self.beta2_k, 0, 0)

        self.beta1 = np.sum(self.beta1_k) + self.c_inf
        self.beta2 = np.sum(self.beta2_k)
        self.beta  = self.beta1 + self.beta2

        if scheme_subtype == 'mCN':
            self.beta1 = self.beta1 + self.beta2
            self.beta2 = 0


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
        self.PrevF3  = np.zeros_like(u)
        self.CurrFH  = np.zeros_like(u)
        self.PrevFH  = np.zeros_like(u)
        self.Residual= np.zeros_like(u)
        self.History = np.zeros_like(u)
        self.Modes = np.zeros([u.shape[0], self.nModes+1])


        ### Initial data
        self.restart()
        self.InitSol[:] = kwargs.get("u_init", self.Model.ArrInitSol) 
        self.CurrSol[:] = self.InitSol[:]
        self.PrevSol[:] = self.InitSol[:]
        self.CurrF[:]   = self.Model.update_NonLinearPart(self.CurrSol, time=self.CurrTime)
        self.PrevF[:]   = self.CurrF[:]
        self.Modes[:,0] = self.Model.multMass(self.InitSol)
        self.History[:] = self.Modes @ self.gamma_k
        
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
        self.CurrF[:]   = self.Model.update_NonLinearPart(self.CurrSol, time=self.CurrTime)
        self.Modes[:]   = self.gamma_k*self.Modes - (self.beta1_k*self.CurrF[:,None] + self.beta2_k*self.PrevF[:,None])
        self.History[:] = self.Modes @ self.gamma_k
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
            # CODE HERE
            pass


#############################################################################################









#############################################################################################
#
#   Fractional Time-Stepper class using Quadrature (NOT IMPLEMENTED)
#
#############################################################################################

class FracTS_Quad(BasicTS):

    def __init__(self, Model, **kwargs):
        super().__init__(Model, **kwargs)

        # CODE HERE

    


    #----------------------------------------------------------------------------------------
    #   Initialize the time integration
    #----------------------------------------------------------------------------------------
    def init_computation(self, MaxTimeStep):

        ### Initialize vectors

        ### (dolfin vectors)
        # TO MODIFY
        y = self.Model.Solution
        self.InitSol = [ dl.Vector(y[i].vector()) for i in range(self.Neq) ]
        self.CurrSol = [ dl.Vector(y[i].vector()) for i in range(self.Neq) ]
        self.CurrInt = [ dl.Vector(y[i].vector()) for i in range(self.Neq) ]
        self.PrevInt = [ dl.Vector(y[i].vector()) for i in range(self.Neq) ]
        self.RHS     = [ dl.Vector(y[i].vector()) for i in range(self.Neq) ]

        ### (numpy arrays)
        # TO MODIFY
        self.Modes = [ np.zeros([self.ndofs_space, len(self.c[i])]) for i in range(self.Neq) ]
        self.GlobalVec = np.zeros(self.ndofs_total)


        ### Precompute linear operators
        # TO MODIFY
        dt, th = self.dt, self.theta
        M  = self.Model.MassMatrix
        L  = self.Model.StiffMatrix
        BC = self.Model.BCs
        self.LinSolver = [0]*self.Neq
        for i in range(self.Neq):
            self.LinSolver[i] = self.set_linSolver()
            A = M[i] + dt*th*self.beta[i]*L[i]
            if BC[i]: BC[i].apply(A)            
            self.LinSolver[i].set_operator(A)

        ### Initial data
        self.Time = self.InitTime
        for i in range(self.Neq):
            self.InitSol[i][:] = self.Model.IC[i].vector()

        ### Initialize QoIs
        self.QoIs = np.zeros([self.Neq, MaxTimeStep+1])
        self.QoIs[:,0] = self.Model.update_QoIs(self.InitSol)


    #----------------------------------------------------------------------------------------
    #   Compute the theta-point of the solution
    #----------------------------------------------------------------------------------------
    def get_theta_point(self):
        # TO MODIFY
        th = self.theta
        y_th = [0]*self.Neq
        for i in range(self.Neq):
            y0, I0, I = self.InitSol[i][:], self.PrevInt[i][:], self.CurrInt[i][:]
            y_th[i] = y0 + ( th*I + (1-th)*I0 )
        return y_th


    #----------------------------------------------------------------------------------------
    #   Run the time integration
    #----------------------------------------------------------------------------------------
    def __call__(self, **kwargs):
        nTimeSteps = kwargs.get("nTimeSteps", self.nTimeSteps)
        self.set_TimeSolver(nTimeSteps)

        MaxTimeStep = kwargs.get("MaxTimeStep", self.nTimeSteps)

        self.init_computation(MaxTimeStep)
        yield self.OutputFormat(self.InitSol)

        ### Time-Stepping
        for self.TimeStep in range(1,MaxTimeStep+1):
            self.Time = self.Time + self.dt

            ### Fixed-Point iteration
            self.FixedPointIter = 0
            y = self.GlobalVec
            y = optimize.fixed_point(self.LinearSolve, y, xtol = 1e-13, method='iteration')
            print('Time step {0:d}, iter {1:d}'.format(self.TimeStep, self.FixedPointIter))

            self.update_Step()

            self.update_QoIs()

            self.UserMonitor(**kwargs)

            yield self.OutputFormat(self.CurrSol)


    #----------------------------------------------------------------------------------------
    #   Linear solver functionfor the Fixed-Point iteration
    #----------------------------------------------------------------------------------------
    def LinearSolve(self, y):
        dt, th = self.dt, self.theta
        y_th = self.get_theta_point()

        F  = self.Model.update_NonLinearPart(y_th)
        L  = self.Model.StiffMatrix
        BC = self.Model.BCs

        y_new = np.zeros_like(y)

        for i in range(self.Neq):
            # CODE HERE
            pass

        self.FixedPointIter += 1

        return y_new


    #----------------------------------------------------------------------------------------
    #   Update the time-step
    #----------------------------------------------------------------------------------------
    def update_Step(self):
        dt, th = self.dt, self.theta
        for i in range(self.Neq):
            # CODE HERE
            pass


    #----------------------------------------------------------------------------------------
    #   Update the quanities of interest
    #----------------------------------------------------------------------------------------
    def update_QoIs(self):
        self.QoIs[:,self.TimeStep] = self.Model.update_QoIs(self.CurrSol)
        return self.QoIs


    #----------------------------------------------------------------------------------------
    #   The routine containing all the user-defined monitoring functions and options
    #----------------------------------------------------------------------------------------
    def UserMonitor(self, **kwargs):
        if kwargs.get("my_flag"):
            # CODE HERE
            pass


#############################################################################################
