import random
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import sys

from source.TimeStepper.FractionalDerivative import FractionalDerivativeRA as FractionalDerivative

###########################################

# Class representing the intial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        # random.seed(2 + MPI.rank(MPI.comm_world))
        random.seed(0)
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0.63 + 0.25*(0.5 - random.random())
        values[1] = 0.0
    def value_shape(self):
        return (2,)

# Class for interfacing with the Newton solver
class CahnHilliardEquation(NonlinearProblem):
    def __init__(self, a, L, Flx, FracDer, Mu0):
        NonlinearProblem.__init__(self)
        self.a = a
        self.L = L
        self.Flx = Flx
        self.FD = FracDer
        self.H = FD.init_history(Mu0)
    def F(self, b, x):
        assemble(self.L, tensor=b)
        b[:] = b - self.H
    def J(self, A, x):
        assemble(self.a, tensor=A)
    def update_hystory(self):
        # vecF1 = -1*assemble(self.Flx[0])
        # vecF2 = -1*assemble(self.Flx[1])
        self.H = FD.update_history(self)


###########################################

# Model parameters
alpha  = 1
lmbda  = 1.0e-01  # surface parameter
dt     = 1.0e-2  # time step
T      = 50*dt
theta  = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

### Fractional Derivtive
scheme = 'Exp'
theta = 1
FD = FractionalDerivative(alpha=alpha, dt=dt, T=T, scheme=scheme, theta=theta)
 
# Form compiler options
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

# Create mesh and build function space
Nx = 2**6
mesh = UnitSquareMesh.create(Nx, Nx, CellType.Type.quadrilateral)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = FunctionSpace(mesh, P1*P1)

# Define trial and test functions
du    = TrialFunction(ME)
q, v  = TestFunctions(ME)

# Define functions
u   = Function(ME)  # current solution
u0  = Function(ME)  # solution from previous converged step

# Split mixed functions
dc, dmu = split(du)
c,  mu  = split(u)
c0, mu0 = split(u0)

# Create intial conditions and interpolate
u_init = InitialConditions(degree=1)
u.interpolate(u_init)
u0.interpolate(u_init)

# Compute the chemical potential df/dc
# c = variable(c)
# c0= variable(c0)
# f    = 100*c**2*(1-c)**2
# dfdc = diff(f, c)
# ### split convex/concave
# Phi_c = lambda x: 100*(3/2 * x**2)
# Phi_e = lambda x: 100*(x**4 - 2*x**3 - 1/2 * x**2)
# phi_c = diff(Phi_c(c), c)
# phi_e = diff(Phi_e(c0), c0)
# # phi_e = diff(Phi_e(c), c)
# phi_c = lambda x: 100*(3 * x)
# phi_e = lambda x: 100*(4*x**3 - 6*x**2 - x)
# phi_c = phi_c(c)
# phi_e = phi_e(c)

# phi1 = lambda x: 100*(4*x*x*x + 2*x)
# phi2 = lambda x: 100*( - 6*x*x)
factor = 1.e2
phi1 = lambda x: factor*(3 * x)
phi2 = lambda x: factor*(4*x*x*x - 6*x*x - x)
Phi = lambda x: factor* x**2 * (1-x)**2

# mu_(n+theta)
# mu_mid = (1.0-theta)*mu0 + theta*mu
b1, b2 = FD.beta1, FD.beta2
mu_mid = b1*mu + b2*mu0

D = 1.e-3


# Weak statement of the equations
F1 = D*dot(grad(mu ), grad(q))*dx
F2 = D*dot(grad(mu0), grad(q))*dx
L0 = c*q*dx + (b1*F1 + b2*F2)
L1 = mu*v*dx - phi1(c)*v*dx - phi2(c)*v*dx - lmbda*dot(grad(c), grad(v))*dx
L = L0 + L1


# Compute directional derivative about u in the direction of du (Jacobian)
a = derivative(L, u, du)

# solver = PETScLUSolver("mumps")
LinSolver = PETScKrylovSolver("gmres", "amg")
LinSolver.parameters["maximum_iterations"] = 1000
LinSolver.parameters["relative_tolerance"] = 1.e-6
LinSolver.parameters["absolute_tolerance"] = 1.e-8
LinSolver.parameters["error_on_nonconvergence"] = True
LinSolver.parameters["nonzero_initial_guess"] = True
LinSolver.parameters["monitor_convergence"] = False

# Create nonlinear problem and Newton solver
problem = CahnHilliardEquation(a, L, [F1,F2], FD, Mu0=assemble(c0*q*dx))
solver = NewtonSolver()
solver.parameters["linear_solver"] = "lu"
# solver.parameters["preconditioner"] = "ilu"
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-6

# solver.parameters["maximum_iterations"] = 1000
# solver.parameters["relative_tolerance"] = 1.e-6
# solver.parameters["absolute_tolerance"] = 1.e-8
# solver.parameters["error_on_nonconvergence"] = True
# solver.parameters["nonzero_initial_guess"] = True
# solver.parameters["monitor_convergence"] = False

# Output file
file = File("output.pvd", "compressed")

### Energy
E = lambda x: assemble( (Phi(x) + 0.5*lmbda*dot(grad(x), grad(x)))*dx ) 
energy = [E(c)]
t_list = [0]

# Step in time
t = 0.0
while (t < T):
    t += dt
    u0.vector()[:] = u.vector()
    solver.solve(problem, u.vector())
    problem.update_hystory()
    file << (u.split()[0], t)
    energy.append(E(c))
    t_list.append(t)

plt.plot(t_list, energy)
plt.show()