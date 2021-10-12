
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

class GrowthModel:

    def __init__(self, **kwargs):

        self.mecha = kwargs.get("mecha", True)
        self.chemo = kwargs.get("chemo", True)

        if self.chemo:
            self.Neq = 3
        else:
            self.Neq = 2

        if self.mecha:
            self.MechFct = self.get_MechFactor(**kwargs)
        else:
            self.MechFct = 1

        self.M_phi = kwargs.get("M_phi", 0)
        self.M_psi = kwargs.get("M_psi", 0)
        self.M_chi = kwargs.get("M_chi", 0)
        self.N_phi = kwargs.get("N_phi", 0)
        self.N_psi = kwargs.get("N_psi", 0)
        self.N_chi = kwargs.get("N_chi", 0)
        self.K_psi = kwargs.get("K_psi", 0)
        self.K_chi = kwargs.get("K_chi", 0)
        self.P_phi = kwargs.get("P_phi", 0)
        self.P_chi = kwargs.get("P_chi", 0)

        self.init_SourceTerms(**kwargs)

        self.init_FiniteElements(**kwargs)

        self.init_BC(**kwargs)

        self.init_IC(**kwargs)

        self.init_Equations(**kwargs)


    #----------------------------------------------------------------------------------------
    #   Compute the factor due to the mechanical coupling
    #----------------------------------------------------------------------------------------
    def get_MechFactor(self, **kwargs):
        c       = kwargs.get("c", 0)
        lamda   = kwargs.get("lamda", 0)
        E       = kwargs.get("E", 0)
        nu      = kwargs.get("nu", 0)        
        G = E/(2*(1+nu))  

        self.MechStruct = {'c':c, 'lamda':lamda, 'E':E, 'nu':nu, 'G':G}

        self.MechFct = c - lamda**2 * (1-2*nu)*(1+nu)/(1-nu)/E

        return self.MechFct


    #----------------------------------------------------------------------------------------
    #   Initialize Source Terms
    #----------------------------------------------------------------------------------------
    def init_SourceTerms(self, **kwargs):
        S_phi = kwargs.get("S_phi", 0)
        S_psi = kwargs.get("S_psi", 0)
        S_chi = kwargs.get("S_chi", 0)

        class SourceTermExpression(dl.UserExpression):
            def __init__(self, func, **kwargs):
                super().__init__(**kwargs)
                self.func = func
                self.t = 0
            def eval(self, value, x):
                value[0] = self.func(self.t, x)
            def value_shape(self):
                return ()

        if callable(S_phi):
            self.S_phi = SourceTermExpression(S_phi)
        else:
            self.S_phi = S_phi

        if callable(S_psi):
            self.S_psi = SourceTermExpression(S_psi)
        else:
            self.S_psi = S_psi

        if callable(S_chi):
            self.S_chi = SourceTermExpression(S_chi)
        else:
            self.S_chi = S_chi

        
    #----------------------------------------------------------------------------------------
    #   Initialize Finite Elements
    #----------------------------------------------------------------------------------------
    def init_FiniteElements(self, **kwargs):

        ### Mesh
        self.shape = kwargs.get("shape")
        self.ndim = len(self.shape)
        self.mesh = generate_Mesh(self.shape, "crossed")

        ### Finite elements spaces
        self.FE_degree = kwargs.get("FE_degree",1)
        FE = dl.FiniteElement("Lagrange", self.mesh.ufl_cell(), self.FE_degree)
        self.Vh = dl.FunctionSpace(self.mesh, FE)
        
        ### Vertices to (dolfin) dofs map
        self.v2d = dl.vertex_to_dof_map(self.Vh)

        
    #----------------------------------------------------------------------------------------
    #   Initialize Boundary Conditions
    #----------------------------------------------------------------------------------------
    def init_BC(self, **kwargs):

        def boundary(x, on_boundary):
            return on_boundary

        ### Homogeneous Neumann is default BC
        BC = kwargs.get("BC", [ ['Neumann', '0'] ]*self.Neq )

        self.BCs = [ None ]*self.Neq
        for i in range(self.Neq):
            if BC[i][0] is 'Dirichlet':
                Value = dl.Expression(BC[i][1], degree=1, t=0)
                self.BCs[i] = dl.DirichletBC(self.Vh, Value, boundary)
        ### Warning: Non-homogenious Neumann is not implemented !
        ### Warning: Time-dependent BCs are not implemented yet !

        
    #----------------------------------------------------------------------------------------
    #   Initialize Initial Conditions
    #----------------------------------------------------------------------------------------
    def init_IC(self, **kwargs):

        ### Zero IC is default
        self.IC = kwargs.get("IC", ['0']*self.Neq )

        IC_expr = [ dl.Expression(IC, degree=1) for IC in self.IC ]
        self.IC = [ dl.interpolate(IC, self.Vh) for IC in IC_expr ]

        
    #----------------------------------------------------------------------------------------
    #   Initialize Equations
    #----------------------------------------------------------------------------------------
    def init_Equations(self, **kwargs):

        ### Auxilary structures
        dx = dl.dx
        uh = dl.TrialFunction(self.Vh)
        vh = dl.TestFunction(self.Vh)

        ### Functions and vectors
        self.Solution = [ dl.Function(self.Vh) for i in range(3) ]
        self.NonLinearPart = [ dl.Vector(self.Solution[i].vector()) for i in range(self.Neq) ]

        ### Linear operators
        L = [0]*3
        L[0] = self.MechFct * self.M_phi * dl.dot( dl.grad(uh), dl.grad(vh) ) * dx
        L[1] =                self.M_psi * dl.dot( dl.grad(uh), dl.grad(vh) ) * dx
        L[2] =                self.M_chi * dl.dot( dl.grad(uh), dl.grad(vh) ) * dx
        self.StiffMatrix = [ dl.assemble(L[i],     tensor=dl.PETScMatrix()) for i in range(self.Neq) ]
        self.MassMatrix  = [ dl.assemble(uh*vh*dx, tensor=dl.PETScMatrix()) for i in range(self.Neq) ]

        ### Non-linear part
        phi, psi, chi = self.Solution
        F = [0]*3
        F[0] = ( self.S_phi + self.N_phi*phi*(1-phi)*psi / (self.K_psi + psi) - self.P_phi*phi*(1-phi)*chi / (self.K_chi + chi) ) * vh*dx
        F[1] = ( self.S_psi - self.N_psi*phi*(1-phi)*psi / (self.K_psi + psi)  ) * vh*dx
        F[2] = ( self.S_chi - self.N_chi*chi - self.P_chi*phi*(1-phi)*chi / (self.K_chi + chi)  ) * vh*dx
        # D = [0]*3
        # D[0] = -self.M_phi * dl.dot( dl.grad(phi), dl.grad(vh) ) * dx * self.MechFct
        # D[1] = -self.M_psi * dl.dot( dl.grad(psi), dl.grad(vh) ) * dx
        # D[2] = -self.M_chi * dl.dot( dl.grad(chi), dl.grad(vh) ) * dx
        self.NonLinearPart_VarForm = [ F[i] for i in range(self.Neq) ]

    #----------------------------------------------------------------------------------------
    ### End of init

        
    #----------------------------------------------------------------------------------------
    #   Update the non-linear part (Right hand side)
    #----------------------------------------------------------------------------------------
    def update_NonLinearPart(self, y, t=0):
        with suppress(AttributeError): self.S_psi.t = t
        with suppress(AttributeError): self.S_chi.t = t
        for i in range(self.Neq):
            self.Solution[i].vector()[:] = y[i][:]
            self.NonLinearPart[i][:] = dl.assemble(self.NonLinearPart_VarForm[i])
        return self.NonLinearPart

    #----------------------------------------------------------------------------------------
    #   Update the quanities of interest
    #----------------------------------------------------------------------------------------
    def update_QoIs(self, y):
        QoIs = [0]*self.Neq
        for i in range(self.Neq):
            u = self.Solution[i]
            u.vector()[:] = y[i][:]
            QoIs[i] = dl.assemble(u*dl.dx)
        return QoIs


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
#	Benchmark boundary conditions
#######################################################################################################

class BenchmarkBoundary(dl.SubDomain):
    
    def inside(self, x, on_boundary):
        return on_boundary #and ( dl.near(x[0], 0) or dl.near(x[0], 1) )


############################################

############################################

def set_linSolver():
	solver = dl.PETScLUSolver("mumps")
	solver = dl.PETScKrylovSolver("cg", "amg")
	# solver = dl.PETScKrylovSolver("gmres", "amg")
	solver.parameters["maximum_iterations"] = 1000
	solver.parameters["relative_tolerance"] = 1.e-6
	solver.parameters["absolute_tolerance"] = 1.e-6
	solver.parameters["error_on_nonconvergence"] = True
	solver.parameters["nonzero_initial_guess"] = True
	solver.parameters["monitor_convergence"] = False
	return solver

################################################################################