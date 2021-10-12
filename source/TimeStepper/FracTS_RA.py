
from math import *
import numpy as np
import dolfin as dl
from scipy import optimize
from scipy.special import gamma, gammainc, beta
import itertools
# import vandermonde
from time import time
import matplotlib.pyplot as plt

import sys
sys.path.append("..")

from .BasicTS import BasicTS
from .RationalApproximation import RationalApproximation_AAA as RationalApproximation
from ..MittagLeffler import ml




#############################################################################################
#
#   Fractional Time-Stepper class using Rational Approximation
#
#############################################################################################

class FracTS_RA(BasicTS):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.alpha = kwargs.get("alpha", 1)
        assert(self.alpha>=0 and self.alpha<=1)

        self.fg_CorrectionTerm = kwargs.get('correction', False)

        self.compute_Coefficients(**kwargs)
        self.compute_AdamsMoultonCoefficients()



    #----------------------------------------------------------------------------------------
    #   Compute the rational approximation coefficients
    #----------------------------------------------------------------------------------------
    def compute_Expansion(self, **kwargs):
        self.RA = RationalApproximation(**kwargs, Zmin=1/self.FinalTime, Zmax=1/self.dt)
        # self.RA = RationalApproximation(**kwargs, Zmin=1/self.FinalTime, Zmax=1/self.dt**2)
        self.nModes = self.RA.nModes
        if self.verbose: print('Number of modes = ', self.RA.nModes)

        c, d, c_inf = self.RA.c, self.RA.d, self.RA.c_inf
        h = self.dt
        # print('Error mCN: ', np.log(np.sum(c/d * (1-np.exp(-d*h)))) / np.log(h)  )
        # # print('Error CN: ', np.sum(c * (1-d*h/2) / (1+d*h/2)))
        # print('Error CN: ', (np.sum(c * 2*h / (1+d*h)))  )
        self.trunc_err = np.sum(c/d * (1-np.exp(-d*h)))

    #----------------------------------------------------------------------------------------
    #   Compute coefficients
    #----------------------------------------------------------------------------------------
    def compute_Coefficients(self, **kwargs):
        self.compute_Expansion(**kwargs)   
        c, d, c_inf = self.RA.c, self.RA.d, self.RA.c_inf
        h = self.dt

        self.Scheme = kwargs.get('scheme', 'RA:mCN')
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

        print('SUM GAMMA_k = ', np.sum(self.gamma_k))

        self.gamma_k = np.insert(self.gamma_k, 0, 1)
        self.beta1_k = np.insert(self.beta1_k, 0, 0)
        self.beta2_k = np.insert(self.beta2_k, 0, 0)

        self.beta1 = np.sum(self.beta1_k) + c_inf
        self.beta2 = np.sum(self.beta2_k)
        self.beta  = self.beta1 + self.beta2

        if scheme_subtype == 'mCN':
            self.beta1 = self.beta1 + self.beta2
            self.beta2 = 0

        ### Initial step (correction term)
        if self.fg_CorrectionTerm:
            h1 = h
            self.beta_ini  = np.sum(c/d*(1-np.exp(-d/h1*h))) * h1**self.alpha + c_inf * h1**self.alpha

            self.beta1_ini = np.sum(c/d**2 *(np.exp(-d) - 1 + d)) * h**self.alpha + c_inf * h**self.alpha
            self.beta2_ini = np.sum(c/d**2 *(1-(1+d)*np.exp(-d))) * h**self.alpha

            self.beta1_ini = c_inf * h**self.alpha
            self.beta2_ini = np.sum(c/d*(1-np.exp(-d))) * h**self.alpha

            self.beta1_k_ini = c/d**2 *(np.exp(-d) - 1 + d) * h**self.alpha
            self.beta2_k_ini = c/d**2 *(1-(1+d)*np.exp(-d)) * h**self.alpha
            pass

    def correction_coef(self, step=0):
        # sigma = 2*self.alpha
        # # sigma = self.alpha
        # ### Correction term 2
        # integrand = self.gamma_k[1:,None]**(1+np.arange(self.TimeStep-1))
        # b1 = np.sum(integrand*self.beta1_k[1:,None], axis=0)
        # b2 = np.sum(integrand*self.beta2_k[1:,None], axis=0)
        # # aux_k = self.beta1_k + (self.beta1_k+self.beta2_k)*np.sum(integrand, axis=-1) + self.gamma_k**(self.TimeStep-1) * self.beta2_k
        # # aux_k = (self.beta1_k+self.beta2_k)*np.sum(integrand, axis=-1)
        # # aux_k = self.gamma_k*aux_k
        # # aux_k = aux_k[1:]
        # # if self.TimeStep<=1: aux_k*=0.
        # # self.CorrCoef = self.CurrTime**self.alpha / gamma(1+self.alpha) - (self.beta + np.sum(aux_k))
        # j = np.arange(self.TimeStep-1)
        # x1 = ( (j+1)/self.TimeStep )**(sigma-self.alpha)
        # x2 = ( j/self.TimeStep )**(sigma-self.alpha)
        # y = self.beta + np.sum(b1*x1[::-1]+b2*x2[::-1])
        # self.CorrCoef = self.CurrTime**self.alpha / gamma(1+self.alpha) * (1+sigma)*beta(1+sigma-self.alpha, 1+self.alpha) - y
        # return self.CorrCoef

        w, lam = self.RA.c, self.RA.d
        m = self.nModes+1
        self.CorrCoefs = np.zeros(m)
        h = self.dt
        tn, tn1 = self.PrevTime, self.CurrTime
        if step==1:
            tn += self.dt
            tn1+= self.dt
            if tn>4*self.dt: return self.CorrCoefs
        elif tn>3*self.dt: return self.CorrCoefs
        N = m
        sigma = np.linspace(self.alpha, 2, N)

        if True: return self.CorrCoefs
        
        A = np.zeros([len(sigma),m])
        b = np.zeros(len(sigma))
        for i in range(len(sigma)):
            A[i,0]  = 1/gamma(1+sigma[i]-self.alpha)
            A[i,1:] = w/lam *tn * ml(-lam*tn, alpha=1, beta=2+sigma[i]-self.alpha)
            # b[i]    = tn1**self.alpha / gamma(1+sigma[i])
            b[i]    = tn1**self.alpha / gamma(1+sigma[i]) - self.beta / gamma(1+sigma[i]-self.alpha) - tn * np.sum( self.gamma_k[1:]*w*ml(-lam*tn, alpha=1, beta=2+sigma[i]-self.alpha) ) ### error

        if tn>0 and False:
            b = A.T @ b
            A = A.T @ A
            y = np.linalg.solve(A, b)
            # y[0]  -= self.beta
            # y[1:] = y[1:]-self.gamma_k[1:]
            self.CorrCoefs = y
        else:
            i=0
            self.CorrCoefs[0] = b[i] / A[i,0]
            # self.CorrCoefs[0] = tn1**self.alpha / gamma(1+sigma[0]) - self.beta - tn * np.sum( self.gamma_k[1:]*w*ml(-lam*tn, alpha=1, beta=2+sigma[0]-self.alpha) )
        return self.CorrCoefs



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
        self.InitF   = np.zeros_like(u)
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
        self.InitF[:]   = self.Model.update_NonLinearPart(self.InitSol, time=self.CurrTime)
        self.CurrF[:]   = self.InitF[:]
        self.PrevF[:]   = self.InitF[:]
        self.Modes[:,0] = self.Model.multMass(self.InitSol)
        # self.Modes[:,1:]= self.CurrF[:,None] * self.compute_initial_modes()[None,:]
        self.History[:] = self.Modes @ self.gamma_k

        # if self.fg_CorrectionTerm:
        #     self.PrevSol_ini = self.PrevSol.copy()
        #     self.PrevF_ini   = self.PrevF.copy()
        #     self.History_ini = self.History.copy()
        #     self.Modes_ini   = self.Modes.copy()

        
        if hasattr(self.Model, 'MatrixUpdate'): self.Model.MatrixUpdate = True

        ### Initialize QoIs
        self.QoIs = [None] * (self.MaxTimeStep+1)
        self.update_QoIs()


    #----------------------------------------------------------------------------------------
    #   Initial modes
    #----------------------------------------------------------------------------------------

    def compute_initial_modes(self):
        N = self.nModes
        x = np.zeros(N)
        A = np.zeros([N,N])
        b = np.zeros(N)

        c, d, c_inf = self.RA.c, self.RA.d, self.RA.c_inf
        alpha, h = self.alpha, self.dt

        # # x = -c_inf * np.exp(d*h)/np.sum(np.exp(d*h))
        # # return x

        # # N = 5
        # t = np.linspace(1.e-1,1,N)[::-1] * h
        # c, d = c[:N], d[:N]
        # A = np.zeros([N,N])
        # b = np.zeros(N)

        # pen = 1.e0
        # reg = 1.e6
        # for i in range(N):
        #     # h=t[i]
        #     b[i] = d[i] * (np.sum(c*(1-np.exp(-(d+d[i])*h))/(d+d[i])) - d[i]**(-alpha) * (1 - gammainc(alpha, d[i]*h))) - pen* c_inf
        #     # b[i] = np.sum(c/d*(1-np.exp(-d*h))) - h**(alpha)/gamma(1+alpha)
        #     for j in range(N):
        #         A[i,j] = d[i]*d[j]*(1-np.exp(-(d[j]+d[i])*h))/(d[j]+d[i]) + pen + reg * np.exp(-d[i]*h) *(i==j)
        #         # A[i,j] = 1-np.exp(-d[j]*h) #+ reg * 1/d[i] * (i==j)  #+ pen

        # x[:N] = np.linalg.solve(A, b)

        # return x

        # t = np.linspace(1.e-1,1,N) * h

        # for i in range(N-1):
        #     b[i] = np.sum(c*np.exp(-d*t[i])) - t[i]**(alpha-1) / gamma(alpha)
        #     for j in range(N):
        #         A[i,j] = np.exp(-d[j]*t[i])

        
        # for j in range(N):
        #     A[-1,j] = 1/d[j]
        # b[-1] = -c_inf


        # x = np.linalg.solve(A, b)
        # x = x/d

        # return x

        # for i in range(N):
        #     b[i] = np.sin(pi*alpha)/pi * gamma(i+1-alpha)
        #     for j in range(m):
        #         A[i,j] = np.exp(-d[j]*h) * (d[j])**i / gamma(i+1) / (d[i])**i

        # x = np.linalg.solve(A, b)

        # x = (x-c)/d

        # x = d/d.min()

        # N = 5
        # x = np.logspace(-6,6,N)

        # def sigma(m,s):
        #     if s<0 or m<0 or s>m:
        #         return 0
        #     elif s==0:
        #         return 1
        #     else:
        #         return sigma(m-1,s)+x[m-1]*sigma(m-1,s-1)

        # def phi(m,s):
        #     if (m==2 and s==1) or (m==2 and s==2):
        #         return 1/(x[1]-x[0])
        #     elif m==s:
        #         return np.prod(1/(x[m-1]-x[:m-1]))
        #     else:
        #         return phi(m-1,s)/(x[m-1]-x[s-1])

        # def psi(n, i, j):
        #     if n==i:
        #         return (-1)**(n+j)
        #     else:
        #         return x[j-1]*psi(n,i+1,j) - (-1)**(i+1+j) * sigma(n,n-i)

        # # invV = np.zeros([N,N])
        # V = np.zeros([N,N])
        # for i in range(N):
        #     for j in range(N):
        #         # invV[i,j] = psi(N,j+1,i+1)*phi(N,i+1)
        #         V[i,j] = x[j]**i

        # b = np.zeros(N)
        # A = np.zeros([N,N])
        # t = np.linspace(0.1, 1, N)*h
        # for i in range(N):
        #     b[i] = np.sum( c*np.exp(-d/h*t[i]) ) * h**(alpha-1) 
        #     b[i] = np.sum( c/d*(1-np.exp(-d)) ) * h**(alpha) 
        #     for j in range(N):
        #         A[i,j] = (1-np.exp(-d[j]*t[i]))/d

        # x = np.linalg.solve(A, b)
        # y = (c - x)/d

        # x = d
        # m = len(x)
        # V = vandermonde.matrix(x).T
        # b = (V @ c) * (1-h**(alpha-1-np.arange(m)))
        # y = vandermonde.solve_transpose(x, b)
        # y = y/d

        # y = c/d * (1 - h**alpha * (1-np.exp(-d))/(1-np.exp(-d*h)) )
        y = c/d * (1 - h**(alpha-1) * np.exp(-d*(1-h)) )

        return y



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

        cc = self.correction_coef()
        self.Model.MatrixUpdate = True
        
        if self.Model.MatrixUpdate:
            M = self.Model.MassMatrix
            K = self.Model.StiffMatrix
            A = M + (self.beta1+cc[0])*K
            if self.fg_CorrectionTerm and self.TimeStep == 1:
                A = M + self.beta1_ini*K          

            if self.verbose:
                print('betas:', self.beta1, self.beta2)
                print('G1:', self.AM1)
                print('G2:', self.AM2[0], self.AM2[1])
                print('G3:', self.AM3[0], self.AM3[1], self.AM3[2])
                print('c_inf:', self.RA.c_inf)
        else:
            A = None


        # self.n_ini=3 #int(1/self.dt/3)
        # if self.fg_CorrectionTerm and self.TimeStep in np.arange(1,self.n_ini+1):
        #     # self.Residual[:] = -self.beta2_ini*self.PrevF + self.History
        #     # self.CurrSol[:]  = self.LSolve(A, self.CurrSol, self.Residual)
        #     # self.Model.MatrixUpdate = True   
        #     self.comput_InitialStep(self.n_ini)
        #     # self.Modes[:,0] = self.Model.multMass(self.InitSol)
        #     # self.Modes[:,1:]= 0*self.Modes[:,1:]
        #     self.Model.MatrixUpdate = True  
        # else:
        self.Residual[:] = -self.beta2*self.PrevF + self.History + 0*(self.beta-self.CurrTime**self.alpha/gamma(self.alpha+1))*self.InitF
        self.CurrSol[:]  = self.LSolve(A, self.CurrSol, self.Residual)
        
        self.NLIter += 1

        return self.CurrSol[:]



    #----------------------------------------------------------------------------------------
    #   Initial step (correction term)
    #----------------------------------------------------------------------------------------
    # def comput_InitialStep(self, n=1):
    #     N = 1000
    #     h_ini = 1/N/n
    #     c, d, c_inf = self.RA.c, self.RA.d, self.RA.c_inf
    #     alpha, h = self.alpha, self.dt
    #     self.gamma_k_ini = np.exp(-d*h_ini)
    #     self.beta1_k_ini = c * np.where(d != 0, (1-(1-np.exp(-d*h_ini))/(d*h_ini))/d, h_ini/2) * h**alpha * n**alpha
    #     self.beta2_k_ini = c * np.where(d != 0, ((1-np.exp(-d*h_ini))/(d*h_ini)-np.exp(-d*h_ini))/d, h_ini/2) * h**alpha * n**alpha

    #     self.gamma_k_ini = np.insert(self.gamma_k_ini, 0, 1)
    #     self.beta1_k_ini = np.insert(self.beta1_k_ini, 0, 0)
    #     self.beta2_k_ini = np.insert(self.beta2_k_ini, 0, 0)

    #     self.beta1_ini = np.sum(self.beta1_k_ini) + c_inf * h**alpha * n**alpha
    #     self.beta2_ini = np.sum(self.beta2_k_ini)
    #     self.beta_ini  = self.beta1 + self.beta2

    #     self.beta1_ini = self.beta1_ini + self.beta2_ini
    #     self.beta2_ini = 0


    #     self.gamma_k_0 = np.exp(-d*h_ini*h*n)
    #     self.beta1_k_0 = c * np.where(d != 0, (1-(1-np.exp(-d*h_ini*h*n))/(d*h_ini*h*n))/d, h_ini*h*n/2)
    #     self.beta2_k_0 = c * np.where(d != 0, ((1-np.exp(-d*h_ini*h*n))/(d*h_ini*h*n)-np.exp(-d*h_ini*h*n))/d, h_ini*h*n/2)


    #     M = self.Model.MassMatrix
    #     K = self.Model.StiffMatrix
    #     for i in range(N):
    #         A = M + self.beta1_ini*K
    #         self.Residual[:] = -self.beta2_ini*self.PrevF_ini + self.History_ini
    #         self.CurrSol[:]  = self.LSolve(A, self.CurrSol, self.Residual)
    #         self.CurrF[:]    = self.Model.update_NonLinearPart(self.CurrSol)
    #         self.Modes_ini[:]    = self.gamma_k_ini*self.Modes_ini - (self.beta1_k_ini*self.CurrF[:,None] + self.beta2_k_ini*self.PrevF_ini[:,None])
    #         self.History_ini[:]  = self.Modes_ini @ self.gamma_k_ini
    #         self.PrevSol_ini[:]  = self.CurrSol[:]
    #         self.PrevF_ini[:]    = self.CurrF[:]
    #         self.Modes[:,1:]  = self.gamma_k_0*self.Modes[:,1:] - (self.beta1_k_0*self.CurrF[:,None] + self.beta2_k_0*self.PrevF_ini[:,None])

            

    #     self.History[:] = self.Modes @ self.gamma_k
    #     self.PrevSol[:] = self.CurrSol[:]
    #     self.PrevF[:]   = self.CurrF[:]

    #     # self.Modes[:,0] = self.Model.multMass(self.InitSol)
    #     # self.Modes[:,1:]= 0*self.Modes[:,1:]
    #     # # self.Modes[:]   = self.gamma_k*self.Modes - (self.beta1_k*self.CurrF[:,None] + self.beta2_k*self.PrevF[:,None])
    #     # # self.History[:] = self.Modes @ self.gamma_k

    #     # self.Model.MatrixUpdate = True   

    #     return self.CurrF[:]


        


    #----------------------------------------------------------------------------------------
    #   Update the fields for the current time-step
    #----------------------------------------------------------------------------------------
    def update_Fields(self):
        cc = np.zeros_like(self.gamma_k)
        cc[1:] = self.CorrCoefs[1:]
        # if (not self.fg_CorrectionTerm) or (self.TimeStep>self.n_ini):
        self.CurrF[:]   = self.Model.update_NonLinearPart(self.CurrSol, time=self.CurrTime)
        # self.Modes[:]   = self.gamma_k*self.Modes - (self.beta1_k*self.CurrF[:,None] + self.beta2_k*self.PrevF[:,None])
        self.Modes[:]   = (self.gamma_k+cc)*self.Modes - ((self.beta1_k+0*self.beta2_k)*self.CurrF[:,None] + self.beta2_k*self.PrevF[:,None] - 0*(self.beta1_k+self.beta2_k)*self.InitF[:,None])
        cc[1:] = self.correction_coef(step=1)[1:]
        self.History[:] = self.Modes @ (self.gamma_k+cc)
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