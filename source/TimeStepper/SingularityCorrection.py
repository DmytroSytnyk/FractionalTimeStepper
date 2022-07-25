
from cmath import exp
import numpy as np
import pyamg
from scipy.linalg import solve
from scipy.special import gamma, hyperu
from matplotlib import pyplot as plt
import dolfin as dl

from source.MittagLeffler import ml


class CorrectionTerm:

    def __init__(self, Problem):
        self.Problem   = Problem
        self.InitModes = np.zeros_like(self.Problem.Modes)
        # for k in range(self.Problem.nModes):
        #     self.InitModes[:,k] = self.Problem.Mu0[:]

        p = 4 ### desired accuracy order of the scheme
        alpha = self.Problem.alpha
        self.nTerms = int(np.ceil(p/alpha))
        n = self.Problem.InitSol.shape[0] ### spatial dimension
        self.coefs  = np.zeros([n, self.nTerms])

        self.compute_Corrector()


    #############################
    def compute_Corrector(self):
        m = self.Problem.nModes
        th_k, w_k = self.Problem.theta_k, self.Problem.w_k
        a_k, b_k, c_k = self.Problem.a_k, self.Problem.b_k, self.Problem.c_k
        a, c = self.Problem.a_bar, self.Problem.c_bar
        h, alpha = self.Problem.dt, self.Problem.alpha
        lmbda_k = th_k/(1-th_k)

        k = np.arange(self.nTerms)
        q = gamma(1+alpha*k)/gamma(1+alpha*(k+1))
        self.q = q


        mdl = self.Problem.Model
        FE = dl.FiniteElement("Lagrange", mdl.mesh.ufl_cell(), self.nTerms)
        Wh = dl.FunctionSpace(mdl.mesh, FE)


        sol_fun = dl.Function(mdl.Vh)
        rhs_fun = dl.Function(mdl.Vh)
        sol_vec = sol_fun.vector()
        rhs_vec = rhs_fun.vector()
        M = 1*mdl.MassMatrix
        K = 1*mdl.StiffMatrix
        if mdl.BC:
            mdl.BC.apply(M)
        #     mdl.BC.apply(M, mdl.VecRHS)
        MSolver = set_linSolver()
        MSolver.set_operator(M)
        # u0 = self.Problem.InitSol
        xx = np.linspace(0,1,1000)
        u0 = np.sin(np.pi*xx)


        ### Set operators
        dx = dl.dx
        uh = dl.TrialFunction(mdl.Vh)
        vh = dl.TestFunction(mdl.Vh)
        Delta = lambda v: dl.div(dl.grad(v))
        self.uh = uh
        # K = [dl.PETScMatrix() for ]*self.nTerms
        # for i in k:
        #     uh = Delta(uh)
        #     K[i] = dl.dot(uh, vh) * dx
        #     K[i] = self.factor * K[i]
        # self.MassMatrix = dl.PETScMatrix()
        # self.StiffMatrix= dl.PETScMatrix()
        # K_form = dl.Constant(1/xx[1]/(4/xx[1]**2*np.sin(np.pi/2*xx[1])**2)) * dl.dot( dl.grad(uh), dl.grad(vh) ) * dx
        # K = dl.assemble(K_form)
        m = mdl.lumped_mass[1]



        def F(u):
            # self.uh = Delta(self.uh)
            # Grad_form = mdl.factor * dl.dot(self.uh, vh) * dx
            # dl.assemble(Grad_form, tensor=K)
            # u0 = self.Problem.InitSol
            # u0 = np.sin(np.pi*xx)
            # v = (u[:-2] + 2*u[1:-1] + u[2:]) / 4
            # v = np.insert(v, 0, 0)
            # v = np.append(v, 0)
            v = u
            rhs_vec.set_local(v)
            Ku = (K * rhs_vec) / m
            if mdl.BC: mdl.BC.apply(Ku)
            MKu = Ku.get_local()
            for i in range(100):
                MKu = (MKu[:-2] + 2*MKu[1:-1] + MKu[2:]) / 4
                MKu = np.insert(MKu, 0, 0)
                MKu = np.append(MKu, 0)
            # MKv = M * Kv
            # for i in range(50):
            #     MKv = M * (MKv / m) 
            # Ku = u[:-2] - 2*u[1:-1] + u[2:]
            # Ku = np.insert(Ku, 0, 0)
            # Ku = np.append(Ku, 0)
            # u  = Ku / xx[1]**2
            # Ku = u[:-2] - 2*u[1:-1] + u[2:]
            # Ku = np.insert(Ku, 0, 0)
            # Ku = np.append(Ku, 0)
            # u  = Ku / xx[1]**2
            # Ku = u[:-2] - 2*u[1:-1] + u[2:]
            # Ku = np.insert(Ku, 0, 0)
            # Ku = np.append(Ku, 0)
            # u  = Ku / xx[1]**2
            # rhs_vec[:] = dl.interpolate(rhs_fun, mdl.Vh).vector().get_local()
            y = -1 * MKu #.get_local()  #/ xx[1]**2 / (4/xx[1]**2*np.sin(np.pi/2*xx[1])**2)
            # y[0] = y[-1] = 0.
            # if mdl.BC:
            #     mdl.BC.apply(rhs_vec)
            # rhs_vec.set_local(y)
            # MSolver.solve(sol_vec, rhs_vec)
            # y[0] = y[-1] = 0.
            # y = -1*sol_vec.get_local()
            return y


        u0 = self.Problem.InitSol
        # u0 = self.Problem.Model.multMass(self.Problem.InitSol)
        # F0 = -1*self.Problem.Model.update_NonLinearPart(u0, time=0)
        F0 = F(u0)


        x = self.coefs
        x[:,0] = F0
        for i in k[1:]:
            # Fx = -1*self.Problem.Model.update_NonLinearPart(x[:,i-1], time=0)
            # x[:,i] = q[i-1] * F(x[:,i-1])
            x[:,i] = F(x[:,i-1])

        for i in k:
            x[:,i] = self.Problem.Model.multMass(x[:,i])

        
    def CorrectorF(self):
        alpha, x_k, q_k = self.Problem.alpha, self.coefs, self.q
        t = self.Problem.CurrTime
        k = np.arange(self.nTerms)
        y = np.sum(x_k * t**(alpha*k), axis=-1)
        return y
    
    def CorrectorResidual(self):
        alpha, x_k, q_k = self.Problem.alpha, self.coefs, self.q
        t = self.Problem.CurrTime
        k = np.arange(self.nTerms)
        y = np.sum(q_k * x_k * t**(alpha*(k+1)), axis=-1)
        return y


    def CorrectorModes(self):
        # x_k = self.InitModes
        # return x_k
        return 0

    def init_Modes(self):
        # self.compute_Corrector()
        self.Problem.Modes[:] = 0 #self.InitModes[:]
        # self.test()


    #----------------------------------------------------------------------------------------
    #   Gradients
    #----------------------------------------------------------------------------------------
    # def get_Grads(self, N):


    #     ### Auxilary structures
    #     dx = dl.dx
    #     uh = dl.TrialFunction(self.Vh)
    #     vh = dl.TestFunction(self.Vh)

    #     for i in range(N):
    #     DF1 = dl.derivative(self.form_F, self.Solution, uh)






################################################################
# Corrector of type hyperu
################################################################

class CorrectionTerm_hyperu:

    def __init__(self, Problem):
        self.Problem = Problem

        self.a_k_backup   = self.Problem.a_k.copy()
        self.b_k_backup   = self.Problem.b_k.copy()
        self.c_k_backup   = self.Problem.c_k.copy()
        self.a_bar_backup = self.Problem.a_bar.copy()
        self.c_bar_backup = self.Problem.c_bar.copy()


    def compute_Correction(self, t):
        m = self.Problem.nModes
        th_k, w_k, b_k = self.Problem.theta_k, self.Problem.w_k, self.b_k_backup
        a, c = self.a_bar_backup, self.c_bar_backup
        h, alpha = self.Problem.dt, self.Problem.alpha
        A = np.zeros([m,m])
        y = np.zeros([m,1])

        betas = np.linspace(0.1,1,m+1)[1:]

        for i in range(m):
            beta = betas[i]
            H = 0
            for j in range(m):
                th = th_k[j]
                lmda = th/(1-th)
                v = ml(-lmda*t, alpha=1, beta=2+beta)
                A[i,j] = 1*v

                vn= 1/(1-th) * (1-h/t)**(1+beta) * t * ml(-lmda*(t-h), alpha=1, beta=2+beta)
                H += w_k[j] * b_k[j] * vn

            F1 = 1 / gamma(1+beta)
            F2 = (1-h/t)**beta / gamma(1+beta)
            u  = t**alpha / gamma(1+beta+alpha)

            y[i] = H + a*F1 + c*F2 - u

        # x = solve(A, y)

        Solver = pyamg.ruge_stuben_solver(A)
        # Solver = pyamg.smoothed_aggregation_solver(A)
        x = Solver.solve(y)

        print( "Residual: ", np.abs(A @ x - y).max() )

        x = x * (1-th_k) / w_k / t

        x = x.flatten()
        return x

    def update(self):
        x_k = 0
        if self.Problem.TimeStep < 2:
            t = self.Problem.CurrTime
            x_k = self.compute_Correction(t)
        self.Problem.a_k   = self.a_k_backup / (1+x_k)
        self.Problem.b_k   = self.b_k_backup / (1+x_k)
        self.Problem.c_k   = self.c_k_backup / (1+x_k)
        self.Problem.a_bar = np.sum(self.Problem.w_k * self.Problem.a_k)
        self.Problem.c_bar = np.sum(self.Problem.w_k * self.Problem.c_k)

    
    def __call__(self):
        self.update()
        alpha = self.Problem.alpha
        t = self.Problem.CurrTime
        U = hyperu(1-alpha, 1-alpha, t) / gamma(alpha)
        return -1 * U * self.F0


    def init_Modes(self):
        self.F0 = self.Problem.Model.update_NonLinearPart(self.Problem.InitSol, time=self.Problem.InitTime)
        self.Problem.Modes[:] = self.F0.reshape([-1,1]) * np.ones([1,self.Problem.nModes])



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