

from math import *
import numpy as np
import dolfin as dl
from scipy import optimize
import copy
from time import time
import matplotlib.pyplot as plt
from contextlib import suppress

import sys
sys.path.append("/home/khristen/Projects/FDE/code/")
sys.path.append("..")

from .BasicModel import BasicModelFEM
from ..MittagLeffler import ml



#######################################################################################################
#
#	Class constructing andcontaining the model operators
#
#######################################################################################################

class DiffusionModel1D(BasicModelFEM):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_Equations(**kwargs)
        
    #----------------------------------------------------------------------------------------
    #   Initialize Equations
    #----------------------------------------------------------------------------------------
    def init_Equations(self, **kwargs):
        self.factor = kwargs.get('stiff', 1) ### extra stiffness factor

        ### Auxilary structures
        dx = dl.dx
        uh = dl.TrialFunction(self.Vh)
        vh = dl.TestFunction(self.Vh)

        ### Functions and vectors
        self.Solution = dl.Function(self.Vh)
        self.VecSol = self.Solution.vector()
        self.VecRHS = dl.Vector(self.VecSol)
        self.VecF   = dl.Vector(self.VecSol)

        ### Arrays (external use)
        self.ArrSol = self.Solution.vector().get_local()
        self.ArrInitSol = self.IC.vector().get_local()

        ### Set operators
        Mass = (uh * vh)*dx
        # Diff = 1/pi**2 * dl.dot( dl.grad(uh), dl.grad(vh) ) * dx
        Diff = dl.dot( dl.grad(uh), dl.grad(vh) ) * dx
        # Diff = Mass
        Diff = self.factor * Diff
        self.MassMatrix = dl.PETScMatrix()
        self.StiffMatrix= dl.PETScMatrix() 
        dl.assemble(Mass, tensor=self.MassMatrix)
        dl.assemble(Diff, tensor=self.StiffMatrix)
        


    #----------------------------------------------------------------------------------------
    #   Update the non-linear part (Right hand side)
    #----------------------------------------------------------------------------------------
    def update_NonLinearPart(self, y, time=0):
        ### Stiffness is constant - no update
        self.VecSol[:] = y
        self.VecF[:] = self.StiffMatrix * self.VecSol

        ### Source term
        if self.Source:
            with suppress(AttributeError):
                self.SourceExpr.t = time
                self.Source = dl.assemble(self.SourceTerm)
            self.VecF[:] -= self.Source

        return self.VecF[:]


    #----------------------------------------------------------------------------------------
    #   Non-Linear solver
    #----------------------------------------------------------------------------------------
    def NLSolve(self, LSolve, y):        
        return LSolve(y)


    #----------------------------------------------------------------------------------------
    #   Linear solver
    #----------------------------------------------------------------------------------------
    def LSolve(self, A, u, rhs):
        self.VecSol[:] = u
        self.VecRHS[:] = rhs
        if self.BC: self.BC.apply(self.VecRHS)
        if self.MatrixUpdate and (A is not None):
            if self.BC: self.BC.apply(A)
            self.LinSolver.set_operator(A)
            # self.LinSolver = dl.PETScLUSolver(A, method="mumps")
            self.MatrixUpdate = False
        
        self.LinSolver.solve(self.VecSol, self.VecRHS)
        
        return self.VecSol[:]


    #----------------------------------------------------------------------------------------
    #   Norm
    #----------------------------------------------------------------------------------------
    def norm(self, u):
        f = dl.Function(self.Vh)
        f.vector().set_local(u[:])
        sqr_norm_f = dl.assemble(f*f*dl.dx)
        norm_f = np.sqrt(sqr_norm_f)
        return norm_f


    #----------------------------------------------------------------------------------------
    #   Analytical solution: only for IC = sin(x)
    #----------------------------------------------------------------------------------------
    def ReferenceSolution(self, y, t=0):
        Phi = ml(-pi**2 *t**self.alpha, alpha=self.alpha)
        # nu = self.alpha
        # Phi0 = ml(-(pi**2*t)**nu, alpha=nu)
        # if self.Source:
        #     Phi1 = t**nu * ml(-t**nu, alpha=nu, beta=nu+1)
        #     Phi  = Phi0 + Phi1
        # else:
        #     Phi = Phi0
        return Phi*self.ArrInitSol