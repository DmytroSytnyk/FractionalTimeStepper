

from math import *
import numpy as np
import dolfin as dl
from scipy import optimize
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

class SimpleScalarModel:

    def __init__(self, **kwargs):
        self.alpha = kwargs.get('alpha', 1)  
        self.Stiff = kwargs.get('stiff', 1)
        self.ndofs = kwargs.get('ndofs', 1)  
        self.shape = [ self.ndofs ]

        self.ArrSol = np.zeros([self.ndofs], dtype=np.float)
        self.ArrInitSol = kwargs.get("IC", '0' ) ### Zero IC is default

        self.MassMatrix  = 1
        self.StiffMatrix = kwargs.get('stiff', 1)
        self.MatrixUpdate = False

        self.Source = False
        


    #----------------------------------------------------------------------------------------
    #   Update the non-linear part (Right hand side)
    #----------------------------------------------------------------------------------------
    def update_NonLinearPart(self, y, time=0):
        return self.StiffMatrix * y


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
        if nu>0:
            Phi0 = ml(-t**nu * a, alpha=nu)
            if self.Source:
                Phi1 = t**nu * a * ml(-t**nu * a, alpha=nu, beta=nu+1)
                Phi  = Phi0 + Phi1
            else:
                Phi = Phi0
            return Phi*self.ArrInitSol
        if nu==0:
            return self.ArrInitSol/(1+a)


