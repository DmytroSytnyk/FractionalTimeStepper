

from math import *
import numpy as np
import dolfin as dl
from scipy import optimize
import copy
from time import time
import matplotlib.pyplot as plt
from contextlib import suppress



#######################################################################################################
#
#	Class constructing andcontaining the model operators
#
#######################################################################################################

class BasicModelFEM:

    def __init__(self, **kwargs):
        self.alpha = kwargs.get('alpha', 1)  

        self.init_FiniteElements(**kwargs)

        self.init_SourceTerms(**kwargs)

        self.init_BC(**kwargs)

        self.init_IC(**kwargs)

        self.init_Equations(**kwargs)

        self.init_Solver(**kwargs)

        
    #----------------------------------------------------------------------------------------
    #   Initialize Finite Elements
    #----------------------------------------------------------------------------------------
    def init_FiniteElements(self, **kwargs):

        ### Mesh
        self.shape = kwargs.get("shape")
        self.ndim = len(self.shape)
        shape1 = [N-1 for N in self.shape]
        self.mesh = generate_Mesh(shape1) #, "crossed")

        ### Finite elements spaces
        self.FE_degree = kwargs.get("FE_degree", 1)
        FE = dl.FiniteElement("Lagrange", self.mesh.ufl_cell(), self.FE_degree)
        self.Vh = dl.FunctionSpace(self.mesh, FE)
        
        ### Vertices to (dolfin) dofs map
        self.v2d = dl.vertex_to_dof_map(self.Vh)


    #----------------------------------------------------------------------------------------
    #   Initialize Source Terms
    #----------------------------------------------------------------------------------------
    def init_SourceTerms(self, **kwargs):
        src = kwargs.get("source", False ) ### no source by default
        # src_expr = dl.Expression(src, degree=1)
        # self.Source = dl.interpolate(src_expr, self.Vh)

        if src not in (False, 0, '0'):
            self.SourceExpr = SourceTermExpression(src) if callable(src) else dl.Expression(src, degree=1)
            vh = dl.TestFunction(self.Vh)
            self.SourceTerm = self.SourceExpr*vh*dl.dx
            self.Source = dl.assemble(self.SourceTerm)
        else:
            self.Source = False

        
    #----------------------------------------------------------------------------------------
    #   Initialize Boundary Conditions
    #----------------------------------------------------------------------------------------
    def init_BC(self, **kwargs):

        def boundary(x, on_boundary):
            return on_boundary

        ### Homogeneous Neumann is default BC
        BC = kwargs.get("BC", ['Neumann', '0'] )

        if BC[0] is 'Dirichlet':
            Value = dl.Expression(BC[1], degree=1, t=0)
            self.BC = dl.DirichletBC(self.Vh, Value, boundary)
        else:
            self.BC = None
        ### Warning: Non-homogenious Neumann is not implemented !
        ### Warning: Time-dependent BCs are not implemented yet !

        
    #----------------------------------------------------------------------------------------
    #   Initialize Initial Conditions
    #----------------------------------------------------------------------------------------
    def init_IC(self, **kwargs):
        IC = kwargs.get("IC", '0' ) ### Zero IC is default
        IC_expr = dl.Expression(IC, degree=1)
        self.IC = dl.interpolate(IC_expr, self.Vh)

        
    #----------------------------------------------------------------------------------------
    #   Initialize Equations
    #----------------------------------------------------------------------------------------
    def init_Equations(self, **kwargs):

        ### Auxilary structures
        dx = dl.dx
        uh = dl.TrialFunction(self.Vh)
        vh = dl.TestFunction(self.Vh)

        ### Functions and vectors
        self.Solution = dl.Function(self.Vh)
        self.VecSol = self.Solution.vector()
        self.VecRHS = dl.Vector(self.VecSol)
        self.VecF   = dl.Vector(self.VecSol)

        ### Set operators
        self.MassMatrix = None
        self.StiffMatrix = None

        
    #----------------------------------------------------------------------------------------
    #   Initialize Linear Solver
    #----------------------------------------------------------------------------------------
    def init_Solver(self, **kwargs):
        self.LinSolver = set_linSolver()
        self.MatrixUpdate = True


    #----------------------------------------------------------------------------------------
    ### End of init


    #----------------------------------------------------------------------------------------
    #   Mass matrix multiplication
    #----------------------------------------------------------------------------------------
    def multMass(self, y):
        self.VecSol[:] = y
        self.VecSol[:] = self.MassMatrix * self.VecSol
        return self.VecSol.get_local()[:]

    #----------------------------------------------------------------------------------------
    #   Update the non-linear part (Right hand side)
    #----------------------------------------------------------------------------------------
    def update_NonLinearPart(self, y, time=0):
        pass

    #----------------------------------------------------------------------------------------
    #   Non-Linear solver
    #----------------------------------------------------------------------------------------
    def NLSolve(self, LSolve, y):        
        pass

    #----------------------------------------------------------------------------------------
    #   Linear solver
    #----------------------------------------------------------------------------------------
    def LSolve(self, A, rhs):
        pass



#######################################################################################################
#	Mesh
#######################################################################################################

def generate_Mesh(shape, *args, **kwargs):
    ndim = len(shape)
    if ndim == 1:
        mesh = dl.UnitIntervalMesh(shape[0], *args, **kwargs)
    elif ndim == 2:
        mesh = dl.UnitSquareMesh(shape[0], shape[1], *args, **kwargs)
    elif ndim == 3:
        mesh = dl.UnitCubeMesh.create(shape[0], shape[1], shape[2], dl.CellType.Type.hexahedron, *args, **kwargs)
    else:
        raise Exception("The case of Dimension={0:d} is not inplemented!".format(ndim))
    return mesh


#######################################################################################################
#	Periodic boundary conditions
#######################################################################################################

class PeriodicBoundary(dl.SubDomain):

	def __init__(self, ndim):
		dl.SubDomain.__init__(self)
		self.ndim = ndim

	def inside(self, x, on_boundary):
		return bool( 	any([ dl.near(x[j], 0) for j in range(self.ndim) ]) and 
					not any([ dl.near(x[j], 1) for j in range(self.ndim) ]) and 
						on_boundary
					)

	def map(self, x, y):
		for j in range(self.ndim):
			if dl.near(x[j], 1):
				y[j] = x[j] - 1
			else:
				y[j] = x[j]

#######################################################################################################
#	Other boundary conditions
#######################################################################################################

class BenchmarkBoundary(dl.SubDomain):
    
    def inside(self, x, on_boundary):
        return on_boundary #and ( dl.near(x[0], 0) or dl.near(x[0], 1) )

                
#######################################################################################################
#	Source term expression class
#######################################################################################################

class SourceTermExpression(dl.UserExpression):
    def __init__(self, func, **kwargs):
        super().__init__(**kwargs)
        self.func = func
        self.t = 0
    def eval(self, value, x):
        value[0] = self.func(self.t, x)
    def value_shape(self):
        return ()


#######################################################################################################
#	Set default linear solver
#######################################################################################################

def set_linSolver():
	# solver = dl.PETScLUSolver("mumps")
	# solver = dl.PETScKrylovSolver("bicgstab", "amg")
	# solver = dl.PETScKrylovSolver("gmres", "amg")
	solver = dl.PETScKrylovSolver("cg", "ilu")
	solver.parameters["maximum_iterations"] = 1000
	solver.parameters["relative_tolerance"] = 1.e-6
	solver.parameters["absolute_tolerance"] = 1.e-6
	solver.parameters["error_on_nonconvergence"] = True
	solver.parameters["nonzero_initial_guess"] = False
	solver.parameters["monitor_convergence"] = False
	return solver

################################################################################