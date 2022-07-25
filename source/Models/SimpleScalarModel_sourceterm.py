

from math import *
import numpy as np
import dolfin as dl
from scipy import optimize
from scipy.special import gamma
import copy
from time import time
import matplotlib.pyplot as plt
from contextlib import suppress
import os, sys

from source.MittagLeffler import ml





#######################################################################################################
#
#	Class constructing and containing the model operators
#
#######################################################################################################

class SimpleScalarModel_sourceterm:

    def __init__(self, **kwargs):
        self.alpha = kwargs.get('alpha', 1)  
        self.Stiff = kwargs.get('stiff', 1)
        self.ndofs = kwargs.get('ndofs', 1)
        self.shape = [ self.ndofs ]

        self.ArrSol = np.zeros([self.ndofs], dtype=np.float)
        self.ArrInitSol = kwargs.get("IC", 0 ) ### Zero IC is default

        self.MassMatrix  = 1
        self.StiffMatrix = kwargs.get('stiff', 1)
        self.MatrixUpdate = False

        self.powers = kwargs.get('source_powers', 1) 
        self.coefs  = kwargs.get('source_coefs',  1)


    def source(self, t):
        return np.sum(self.coefs * t**self.powers  / gamma(1+self.powers))
        # return 1 


    #----------------------------------------------------------------------------------------
    #   Update the non-linear part (Right hand side)
    #----------------------------------------------------------------------------------------
    def update_NonLinearPart(self, y, time=0):
        return self.StiffMatrix * y - self.source(time)


    #----------------------------------------------------------------------------------------
    #   Non-Linear solver
    #----------------------------------------------------------------------------------------
    def NLSolve(self, LSolve, y):        
        return LSolve(y)


    #----------------------------------------------------------------------------------------
    #   Linear solver
    #----------------------------------------------------------------------------------------
    def LSolve(self, A, u, rhs):
        if self.MatrixUpdate:
            self.MatrixA = A
            self.MatrixUpdate = False
        return rhs / self.MatrixA


    #----------------------------------------------------------------------------------------
    #   Mass matrix multiplication
    #----------------------------------------------------------------------------------------
    def multMass(self, y):
        return y


    #----------------------------------------------------------------------------------------
    #   Norm
    #----------------------------------------------------------------------------------------
    def norm(self, u):
        return np.abs(u)


    #----------------------------------------------------------------------------------------
    #   Reference solution
    #----------------------------------------------------------------------------------------
    def ReferenceSolution(self, y, t=0):
        nu = self.alpha
        a  = self.StiffMatrix
        u0 = self.ArrInitSol
        if nu>0:
            Phi = ml(-t**nu * a, alpha=nu)*u0
            for i, beta in enumerate(self.powers):
                Phi += self.coefs[i] * t**(nu+beta) * ml(-t**nu * a, alpha=nu, beta=nu+beta+1)  ### sum-of-powers source
            # Phi += t**nu * ml(-t**nu * a, alpha=nu, beta=nu+1) ### constant source
            # Phi += t**nu * ml(-t**nu * a, alpha=nu, beta=nu+1)  ### constant source
            return Phi
        if nu==0:
            return self.ArrInitSol/(1+a)


