

from math import *
import numpy as np
import dolfin as dl
from scipy import optimize
import copy
from time import time
import matplotlib.pyplot as plt
from contextlib import suppress
import sys, os
import scipy.optimize

from .BasicModel import BasicModelFEM
from ..MittagLeffler import ml



#######################################################################################################
#
#	Class constructing and containing the model operators
#
#######################################################################################################

class BurgersEquation1D(BasicModelFEM):

    def __init__(self, **kwargs):
        a, b = kwargs.get('param_a', 1), kwargs.get('param_b', 20)
        kwargs['IC'] = f'2*pi*{a}*sin(pi*x[0])/ ({b} + cos(pi*x[0]))'
        self.a, self.b = a, b
        super().__init__(**kwargs)
        # self.init_Equations(**kwargs)
        
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
        self.MassMatrix = dl.PETScMatrix()
        self.StiffMatrix= dl.PETScMatrix() 
        Mass = (uh * vh)*dx
        dl.assemble(Mass, tensor=self.MassMatrix)
        # Diff = dl.dot( dl.grad(uh), dl.grad(vh) ) * dx
        # Diff = self.factor * Diff
        # dl.assemble(Diff, tensor=self.StiffMatrix)

        self.form_NL   = dl.inner( self.Solution * self.Solution.dx(0), vh ) * dx
        self.form_Diff = dl.inner( dl.grad(self.Solution), dl.grad(vh) ) * dx
        self.form_F    = self.form_NL + self.a * self.form_Diff
        self.form_DF   = dl.derivative(self.form_F, self.Solution, uh)

        # self.update_NonLinearPart(self.ArrInitSol)

        ### sin(x) and cos(x)
        sin_expr = dl.Expression('sin(pi*x[0])', degree=1)
        cos_expr = dl.Expression('cos(pi*x[0])', degree=1)
        self.sinpix = dl.interpolate(sin_expr, self.Vh).vector().get_local()
        self.cospix = dl.interpolate(cos_expr, self.Vh).vector().get_local()
        


    #----------------------------------------------------------------------------------------
    #   Update the non-linear part (Right hand side)
    #----------------------------------------------------------------------------------------
    def update_NonLinearPart(self, y, time=0):        
        self.VecSol[:] = y
        self.VecF[:]   = dl.assemble(self.form_F)
        dl.assemble(self.form_DF, tensor=self.StiffMatrix)

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
        nitermaxNewton = 20
        eps = 1.e-6
        yold = np.copy(y)
        for iterNewton in range(nitermaxNewton):
            y = LSolve(y)
            if np.linalg.norm(y-yold) < eps*np.linalg.norm(yold): break
            yold[:] = y
        # print(iterNewton, np.linalg.norm(y-yold))
        return y


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
            self.MatrixUpdate = True
        
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
        a, b, alpha = self.a, self.b, self.alpha
        ML = ml(-pi**2 * a * t**alpha, alpha=alpha) 
        RefSol = 2*pi*a*ML*self.sinpix/ (b + ML*self.cospix)
        return RefSol


