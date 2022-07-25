
from math import *
import numpy as np
import dolfin as dl
from scipy import optimize
from .RationalApproximation import RationalApproximation_AAA
from time import time
import matplotlib.pyplot as plt





#############################################################################################
#
#   Basic Time-Stepper class (obsolete)
#
#############################################################################################

class BasicTS:

    def __init__(self, **kwargs):
        self.verbose = kwargs.get("verbose", False)

        self.InitTime  = kwargs.get('InitTime',  0)
        self.FinalTime = kwargs.get('FinalTime', 1)
        assert(self.InitTime < self.FinalTime)
        self.TimeInterval = self.FinalTime - self.InitTime

        self.nTimeSteps = kwargs.get('nTimeSteps', None)
        if self.nTimeSteps is not None:
            self.dt = self.TimeInterval / self.nTimeSteps
        else:
            self.dt = kwargs.get('dt', None)
            self.nTimeSteps = ceil(self.TimeInterval / self.dt)
            assert(self.dt < self.TimeInterval)
        self.MaxTimeStep = self.nTimeSteps

        self.restart()

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
        return self.stop()

    #----------------------------------------------------------------------------------------
    #   Stop criterion
    #----------------------------------------------------------------------------------------
    def stop(self):
        return self.CurrTime > self.FinalTime 


#############################################################################################

#############################################################################################
#
#   Fractional Derivtive class using Rational Approximation
#
#############################################################################################

class FractionalDerivativeRA: #(BasicTS):

    def __init__(self, **kwargs):
        # super().__init__(**kwargs)

        self.alpha = kwargs.get("alpha", 1)
        assert(self.alpha>=0 and self.alpha<=1)

        ### Time-stepper
        self.TS = BasicTS(**kwargs)
        self.h = self.TS.dt
        self.T = self.TS.TimeInterval

        self.set_Scheme(**kwargs)
        self.compute_Expansion(**kwargs)
        self.compute_Coefficients(**kwargs)

        print()
        print('=================================')
        print('Fractional Derivative scheme: RA')
        print('=================================')
        print('alpha:  ', self.alpha)
        print('Scheme: ', self.Scheme)
        print('Number modes:      ', self.nModes)
        print('Number time-steps: ', self.TS.nTimeSteps)
        print()


    #----------------------------------------------------------------------------------------
    #   Set Scheme
    #----------------------------------------------------------------------------------------

    def set_Scheme(self, **kwargs):
        Scheme = kwargs.get("scheme", 'Exponential')
        self.theta = kwargs.get("theta", 1)

        if Scheme in ('theta-scheme', 'theta', 'Std', 'Standard'):
            self.Scheme = 'theta'

        elif Scheme in ('Exp', 'Exponential'):
            self.Scheme = 'Exponential'

        elif Scheme in ('RAstab',):
            self.Scheme = 'RAstab'

        else:
            raise Exception('Wrong scheme {0:s}'.format(Scheme))


    #----------------------------------------------------------------------------------------
    #   Compute the rational approximation
    #----------------------------------------------------------------------------------------
    def compute_Expansion(self, **kwargs):
        self.Method = kwargs.get('Method', 7)
        self.Scheme = kwargs.get('scheme', 'Exponential')
        tol = kwargs.get("tol", 1.e-12)
        nNodes = kwargs.get("nNodes", 100)
        h, T = self.h, self.T
        SpectrumInterval = kwargs.get("SpectrumInterval", [1/T, 1/h])
        Zmin, Zmax = SpectrumInterval

        self.RA = RationalApproximation_AAA(    **kwargs,
                                                Zmin= Zmin, Zmax= Zmax,
                                                fg_Reciprocal=True)
        self.c, self.d = self.RA.c, self.RA.d
        

        self.nModes = self.d.size
        assert(np.all(self.d>=0))

        self.c_inf = 0 ### No residue term by default
        if self.c.size > self.d.size :
            self.c_inf, self.c = self.c[-1], self.c[:-1]


    #----------------------------------------------------------------------------------------
    #   Compute coefficients
    #----------------------------------------------------------------------------------------
    def compute_Coefficients(self, **kwargs):
        if not hasattr(self, 'RA'): self.compute_Expansion(**kwargs)
        h, c, d = self.h, self.c, self.d
        th = self.theta

        if self.Scheme == 'Exponential':
            self.gamma_k = np.exp(-d*h)
            # self.beta_k  = c * np.where(d != 0, (1-np.exp(-d*h))/d, h)
            # if th == 1:
            #     self.beta1_k = self.beta_k
            #     self.beta2_k = 0*self.beta_k
            # elif th == 0.5:
                # self.beta1_k = self.beta_k*th
                # self.beta2_k = self.beta_k*(1-th)
            self.beta1_k = c * np.where(d != 0, (1-(1-np.exp(-d*h))/(d*h))/d, h/2)
            self.beta2_k = c * np.where(d != 0, ((1-np.exp(-d*h))/(d*h)-np.exp(-d*h))/d, h/2)
            # else:
            #     raise Exception()

        elif self.Scheme == 'theta':
            self.gamma_k = (1 - (1-th)*d*h) / (1 + th*d*h)
            self.beta_k  = c*h / (1 + th*d*h)
            self.beta1_k = self.beta_k*th
            self.beta2_k = self.beta_k*(1-th)

        elif self.Scheme == 'RAstab':
            self.gamma_k = np.exp(-d*h)
            self.beta_k  = c * np.where(d != 0, (1-np.exp(-d*h))/d, h)
            self.beta1_k = self.beta_k
            self.beta2_k = 0*self.beta_k

        else:
            raise Exception('Wrong scheme {0:s}'.format(self.Scheme))

        # self.beta  = np.sum(self.beta_k)
        self.beta1 = np.sum(self.beta1_k) + self.c_inf
        self.beta2 = np.sum(self.beta2_k)

        self.gamma_k = np.insert(self.gamma_k, 0, 1)
        # self.beta_k  = np.insert(self.beta_k,  0, 0)
        self.beta1_k = np.insert(self.beta1_k, 0, 0)
        self.beta2_k = np.insert(self.beta2_k, 0, 0)


        ### Adam-Mouton coefficients

        a = self.alpha

        self.G1 = h**a * 1/gamma(a+1)

        self.G2 = [0]*2
        self.G2[0] = h**a * 1/gamma(a+2)
        self.G2[1] = h**a * a/gamma(a+2)

        self.G3 = [0]*3
        self.G3[0] = h**a * ( 1/(2*gamma(a+2)) + 1/gamma(a+3) )
        self.G3[1] = h**a * ( 1/ gamma(a+1) - 2/ gamma(a+3) )
        self.G3[2] = h**a * ( -1/ (2*gamma(a+2)) + 1/ gamma(a+3) )



    #----------------------------------------------------------------------------------------
    #   History init
    #----------------------------------------------------------------------------------------

    def init_history(self, u0):
        self.uk = [0]*(self.nModes + 1)
        self.uk[0] = u0
        H = 0 + self.uk[0]
        return H


    #----------------------------------------------------------------------------------------
    #   History update
    #----------------------------------------------------------------------------------------

    def update_history(self, problem):
        m = self.nModes + 1 
        g, b1, b2 = self.gamma_k, self.beta1_k, self.beta2_k
        uk = self.uk

        F1 = -1*dl.assemble(problem.Flx[0])
        F2 = -1*dl.assemble(problem.Flx[1])
        # S  = -1*dl.assemble(problem.Stab)

        H = 0
        for k in range(m):
            uk[k] = g[k]*uk[k] + b1[k]*F1 + b2[k]*F2 #+ a[k]*S
            H = H + g[k]*uk[k]

        return H



#############################################################################################




#############################################################################################
#
#   Fractional Derivtive class using L1 scheme
#
#############################################################################################

class FractionalDerivativeL1:

    def __init__(self, **kwargs):
        self.verbose = kwargs.get("verbose", False)

        self.alpha = kwargs.get("alpha", 1)
        assert(self.alpha>=0 and self.alpha<=1)

        ### Time-stepper
        self.TS = BasicTS(**kwargs)
        self.h = self.TS.dt
        self.T = self.TS.TimeInterval
        self.theta = 1 #kwargs.get("theta", 1)

        self.compute_Coefficients(**kwargs)

        print('=================================')
        print('Fractional Derivative scheme: L1')
        print('=================================')


    #----------------------------------------------------------------------------------------
    #   Compute coefficients
    #----------------------------------------------------------------------------------------
    def compute_Coefficients(self, **kwargs):
        a, h = self.alpha, self.h
        th = self.theta
        b0 = 1/gamma(2-a)
        self.b = [ b0 ]

        self.beta1 = h**a / b0 * th
        self.beta2 = h**a / b0 * (1-th)


    #----------------------------------------------------------------------------------------
    #   History init
    #----------------------------------------------------------------------------------------

    def init_history(self, u0):
        self.u = [u0]
        H = 0 + self.u[0]
        return H
      

    #----------------------------------------------------------------------------------------
    #   History update
    #----------------------------------------------------------------------------------------

    def update_history(self, problem):
        a = self.alpha        
        j = len(self.b)
        bj = ((j+1)**(1-a) - j**(1-a)) / gamma(2-a)
        uj = dl.assemble(problem.Mu)
        self.b.append(bj)
        self.u.append(uj)

        b, u, n = self.b, self.u, len(self.u)
        H = b[-1]/b[0] * u[0]
        for j in range(1,n):
            H = H - (b[j]-b[j-1])/b[0] * u[n-j]

        return H



#############################################################################################
