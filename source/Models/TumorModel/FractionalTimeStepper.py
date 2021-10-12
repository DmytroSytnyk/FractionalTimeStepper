
from math import *
import numpy as np
import dolfin as dl
from scipy import optimize
from RationalApproximation import compute_RationalApproximation_AAA
from time import time
import matplotlib.pyplot as plt

#############################################################################################
#
#   Basic Fractional Time-Stepper class (abstract)
#
#############################################################################################

class BasicFracTS:

    def __init__(self, Model, InitTime=0, FinalTime=1, nTimeSteps=None, theta=1, **kwargs):

        self.Model = Model
        self.Neq = self.Model.Neq

        self.InitTime = InitTime
        self.FinalTime = FinalTime
        assert(self.InitTime < self.FinalTime)
        self.TimeInterval = self.FinalTime - self.InitTime
        self.set_TimeSolver(nTimeSteps)
        self.theta = theta

        self.domain_shape = self.Model.shape
        self.ndim = len(self.domain_shape)
        self.ndofs_space = np.prod(self.domain_shape)

        self.ndofs_local = np.array([ self.Model.Solution[i].vector().size() for i in range(self.Neq) ])
        self.ndofs_total = np.sum(self.ndofs_local)
        self.dofs = [ np.arange(self.ndofs_local[i]) + np.sum(self.ndofs_local[:i]) for i in range(self.Neq) ]

        lspace = [ np.linspace(0,1,N) for N in self.domain_shape ]
        self.grid_space = np.meshgrid(*lspace, indexing='xy')


    #----------------------------------------------------------------------------------------
    #   Set the time-stepper options
    #----------------------------------------------------------------------------------------
    def set_TimeSolver(self, nTimeSteps):
        if nTimeSteps:
            self.nTimeSteps = nTimeSteps
            self.dt = self.TimeInterval / self.nTimeSteps

    
    #----------------------------------------------------------------------------------------
    #   Set default linear solver
    #----------------------------------------------------------------------------------------
    def set_linSolver(self):
        # solver = dl.PETScLUSolver("mumps")
        # solver = dl.PETScKrylovSolver("gmres", "amg")
        solver = dl.PETScKrylovSolver("cg", "amg")
        solver.parameters["maximum_iterations"] = 1000
        solver.parameters["relative_tolerance"] = 1.e-6
        solver.parameters["absolute_tolerance"] = 1.e-8
        solver.parameters["error_on_nonconvergence"] = True
        solver.parameters["nonzero_initial_guess"] = True
        solver.parameters["monitor_convergence"] = False
        return solver


    #----------------------------------------------------------------------------------------
    #   Output the solution in the geometrically consistent numpy array format
    #   (finaloutput format)
    #----------------------------------------------------------------------------------------
    def OutputFormat(self, y):
        v2d, shp = self.Model.v2d, self.domain_shape
        # return [ y[i][v2d].reshape(shp) for i in range(self.Neq) ]
        return [ y[i][v2d] for i in range(self.Neq) ]

#############################################################################################











#############################################################################################
#
#   Fractional Time-Stepper class using Rational Approximation
#
#############################################################################################

class FracTS_RA(BasicFracTS):

    def __init__(self, Model, **kwargs):
        super().__init__(Model, **kwargs)
        alpha = kwargs.get("alpha", 1)
        self.compute_Coefficients(alpha)

    #----------------------------------------------------------------------------------------
    #   Precompute the rational approximation
    #----------------------------------------------------------------------------------------
    def compute_Coefficients(self, alpha):
        # if np.isscalar(nu):
        #     nu = [nu] + [1]*(self.Neq-1)
        assert(alpha>0 and alpha<=1)
        self.alpha = alpha

        c, d = compute_RationalApproximation_AAA(alpha)

        self.OrderRA = c.size
        print('RA order = ', self.OrderRA)

        assert(np.all(d>=0))

        d_int = np.zeros(1)
        c_int = np.ones(1)

        self.c = [ c_int for i in range(self.Neq) ]
        self.d = [ d_int for i in range(self.Neq) ]

        self.c[0] = c
        self.d[0] = d

        dt, theta = self.dt, self.theta
        self.beta = [ np.sum(self.c[i] / (1 + dt*theta*self.d[i])) for i in range(self.Neq) ]


    #----------------------------------------------------------------------------------------
    #   Initialize the time integration
    #----------------------------------------------------------------------------------------
    def init_computation(self, MaxTimeStep):

        ### Initialize vectors

        ### (dolfin vectors)
        y = self.Model.Solution
        self.InitSol = [ dl.Vector(y[i].vector()) for i in range(self.Neq) ]
        self.CurrSol = [ dl.Vector(y[i].vector()) for i in range(self.Neq) ]
        self.CurrInt = [ dl.Vector(y[i].vector()) for i in range(self.Neq) ]
        self.PrevInt = [ dl.Vector(y[i].vector()) for i in range(self.Neq) ]
        self.RHS     = [ dl.Vector(y[i].vector()) for i in range(self.Neq) ]
        self.thSol   = [ dl.Vector(y[i].vector()) for i in range(self.Neq) ]

        ### (numpy arrays)
        self.Modes = [ np.zeros([self.ndofs_local[i], len(self.c[i])]) for i in range(self.Neq) ]
        self.GlobalVec = np.zeros(self.ndofs_total)


        ### Precompute linear operators
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
            self.CurrSol[i][:] = self.InitSol[i][:]

        ### Initialize QoIs
        self.QoIs = np.zeros([self.Neq, MaxTimeStep+1])
        self.QoIs[:,0] = self.Model.update_QoIs(self.InitSol)


    #----------------------------------------------------------------------------------------
    #   Compute the theta-point of the solution
    #----------------------------------------------------------------------------------------
    def get_theta_point(self):
        th = self.theta
        for i in range(self.Neq):
            y0, I0, I = self.InitSol[i][:], self.PrevInt[i][:], self.CurrInt[i][:]            
            self.thSol[i][:] = y0 + ( th*I + (1-th)*I0 )
        return self.thSol


    #----------------------------------------------------------------------------------------
    #   Run the time integration
    #----------------------------------------------------------------------------------------
    def __call__(self, **kwargs):
        nTimeSteps = kwargs.get("nTimeSteps", self.nTimeSteps)
        self.set_TimeSolver(nTimeSteps)

        MaxTimeStep = kwargs.get("MaxTimeStep", self.nTimeSteps)
        xtol = kwargs.get("FixedPointTol", 1.e-13)

        self.init_computation(MaxTimeStep)
        yield self.OutputFormat(self.InitSol)

        ### Time-Stepping
        for self.TimeStep in range(1,MaxTimeStep+1):
            self.Time = self.Time + self.dt

            ### Fixed-Point iteration
            self.FixedPointIter = 0
            self.GlobalVec = optimize.fixed_point(self.LinearSolve, self.GlobalVec, xtol=xtol, method='iteration')
            print('Time step {0:d}, iter {1:d}'.format(self.TimeStep, self.FixedPointIter))

            self.update_fields()

            self.update_QoIs()

            self.UserMonitor(**kwargs)

            yield self.OutputFormat(self.CurrSol)


    #----------------------------------------------------------------------------------------
    #   Linear solver functionfor the Fixed-Point iteration
    #----------------------------------------------------------------------------------------
    def LinearSolve(self, y):
        dt, th = self.dt, self.theta
        y_th = self.get_theta_point()

        t0 = time()
        F  = self.Model.update_NonLinearPart(y_th, self.Time)
        L  = self.Model.StiffMatrix
        BC = self.Model.BCs
        # print('Update F: ', time()-t0)

        y_new = np.zeros_like(y)

        for i in range(self.Neq):
            u, b = self.CurrInt[i], self.RHS[i]
            y0, u0 = self.InitSol[i], self.PrevInt[i]
            Lu = L[i]*(y0 + (1-th)*u0)
            t0 = time()
            coef = self.c[i] * (1 - (1-th)*dt*self.d[i]) / (1 + th*dt*self.d[i])
            b[:] = self.Modes[i] @ coef  + dt*self.beta[i]*( F[i][:] - Lu[:] )
            # print('Time RHS modes: ', time()-t0)
            t0 = time()
            if BC[i]: BC[i].apply(b)
            self.LinSolver[i].solve(u, b)
            y_new[self.dofs[i]] = u[:]
            # print('Time Solve: ', time()-t0)

        self.FixedPointIter += 1

        return y_new


    #----------------------------------------------------------------------------------------
    #   Update the fields for the current time-step
    #----------------------------------------------------------------------------------------
    def update_fields(self):
        dt, th = self.dt, self.theta
        L = self.Model.StiffMatrix
        y_th = self.get_theta_point()
        for i in range(self.Neq):
            r = self.Model.NonLinearPart[i].get_local() - (L[i]*y_th[i])[:]
            coef = (1 - (1-th)*dt*self.d[i]) / (1 + th*dt*self.d[i])
            self.Modes[i] = coef*self.Modes[i] + dt*r[:,None] / (1 + th*dt*self.d[i][None,:])
            self.CurrSol[i][:] = self.InitSol[i][:] + self.CurrInt[i][:]
            self.PrevInt[i][:] = self.CurrInt[i][:]


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









#############################################################################################
#
#   Fractional Time-Stepper class using Quadrature
#
#############################################################################################

class FracTS_Quad(BasicFracTS):

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