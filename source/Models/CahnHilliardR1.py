import numpy as np
from dolfin import *
from source.TimeStepper import FracTS_RA, FracTS_L1
from .CahnHilliardRUtils import Exporter, IC_FourBubbles

###########################################
# Form compiler options
# are supposed to be defined in the caller script

parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

class CahnHilliardR1(NonlinearProblem):

    def __init__(self, **kwargs):
        NonlinearProblem.__init__(self)
        # Save config before it get's modified
        self.config = kwargs
        # Add class name to config dictionary so that the output data can be easily distinguished
        self.config['Name'] = self.__class__.__name__

        self.verbose = kwargs.get('verbose', False)
        set_log_active(self.verbose)
        # set_log_level(40)

        ### Spatial dimension
        self.ndim = kwargs.get('ndim', 2)
        
        ### Time-stepper
        self.TS = self.set_TimeStepper(**kwargs)
        b1, b2  = self.TS.beta1, self.TS.beta2

        ### Exports
        self.Exporter = Exporter(self, **kwargs)

        ###--------------------------------------------------------

        ### Parameters
        zeros = kwargs.get("zeros", [-1,1])
        self.eps   = kwargs.get('eps', 1.e-1)    # surface parameter
        lmbda = Constant(self.eps**2)

        ### Chemical potential derivative
        if zeros==[0,1]:
            _Phi  = eval(kwargs.get('Phi',  'lambda x: 0.25 * (1-x**2)**2'          ))
            _phi1 = eval(kwargs.get('phi1', 'lambda x: 0.25 * (3 * x)'              ))   ### convex part
            _phi2 = eval(kwargs.get('phi2', 'lambda x: 0.25 * (4*x*x*x - 6*x*x - x)'))   ### concave part
        elif zeros==[-1,1]:
            _Phi  = eval(kwargs.get('Phi',  'lambda x: 0.25 * (1-x**2)**2'    ))
            _phi1 = eval(kwargs.get('phi1', 'lambda x: 0.25 * (8 * x)'        ))  ### convex part
            _phi2 = eval(kwargs.get('phi2', 'lambda x: 0.25 * (4*x**3 - 12*x)'))  ### concave part
        else:
            raise Exception('Don\'t know these zeros.')
        ### String representation of phi1 is needed for linear solver assembly
        sphi1 = kwargs.get('phi1')   
        if not callable(_Phi):
            self.Phi  = lambda x: _Phi
        else:
            self.Phi  = _Phi
        if not callable(_phi1):
            self.phi1 = lambda x: _phi1
        else:
            self.phi1 = _phi1
            sphi1 = sphi1.replace('lambda x:','')
        if not callable(_phi2):
            self.phi2 = lambda x: _phi2
        else:
            self.phi2 = _phi2
        
        ### Mobility
        _M = eval(kwargs.get('M', 'lambda x: (1-x**2)**2'))
        if not callable(_M):
            self.M = lambda x: Constant(_M)
        else:
            self.M = _M

        ### Derivative of mobility
        _Md = eval(kwargs.get('Md', 'lambda x: -4*x*(1-x**2)'))
        if not callable(_Md):
            self.Md = lambda x: Constant(_Md)
        else:
            self.Md = _Md

        ### At the beginning the initial mass is unknown
        self.mass_init = float('nan')

        ### Mesh
        self.Nx   = kwargs.get('Nx', 2**5)    # number of grid points in space (one direction)
        self.ndim = kwargs.get('ndim', 2)
        if self.ndim == 2:
            self.mesh = UnitSquareMesh.create(self.Nx, self.Nx, CellType.Type.quadrilateral)
        elif self.ndim == 3:
            self.mesh = UnitCubeMesh.create(self.Nx, self.Nx, self.Nx, CellType.Type.hexahedron)
        else:
            raise Exception('The dimension is not implemented!')

        ### FEniCS related options
        self._fenics_assembly_pars = dict(form_compiler_parameters = {"quadrature_degree": 1}) 

        
        ### Function space
        fg_PerBC = kwargs.get('PerBC', False)
        if fg_PerBC:
            ### Periodic BCs
            BC = PeriodicBoundary(ndim=self.ndim)
            P1 = FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
            self.V1 = FunctionSpace(self.mesh, P1, constrained_domain=BC)
            self.V2 = FunctionSpace(self.mesh, P1, constrained_domain=BC)
            self.V3 = FunctionSpace(self.mesh, P1, constrained_domain=BC)
            W = FunctionSpace(self.mesh, MixedElement([P1, P1, P1]), constrained_domain=BC)
            self.W = W
        else:
            ### Homogeneous Neumann BCs
            P1 = FiniteElement("Lagrange", self.mesh.ufl_cell(), 1)
            W = FunctionSpace(self.mesh, MixedElement([P1, P1, P1]))
            self.W = W
            self.V1 = W.sub(0).collapse()
            # Note that the function space V1_vec will represent gradients of V1 hence the use of DG0 elements
            # self.V1_vec = VectorFunctionSpace(self.mesh, "DG", 0)
            self.V2 = W.sub(1).collapse()
            self.V3 = W.sub(2).collapse()
        self.v2d = vertex_to_dof_map(self.V1)
        self.dofmap1 = self.W.sub(0).dofmap()
        self.dofmap2 = self.W.sub(1).dofmap()
        self.dofmap3 = self.W.sub(2).dofmap()


        # Define trial and test functions
        du      = TrialFunction(W)
        q, v, _ = TestFunctions(W)

        # Define functions
        u   = Function(W)  # current solution
        u0  = Function(W)  # solution from previous converged step
        u1  = Function(W)  # solution from one step before previous converged step

        # Split mixed functions
        dc, dmu, deta = split(du)
        c,  mu, eta   = split(u)
        c0, mu0, eta0 = split(u0)
        # We only need concentration from two steps before
        c1 = split(u1)[0];

        # Create initial conditions and interpolate
        IC = eval(kwargs.get('IC'))
        ICParameters = kwargs.get('ICParameters',{})
        if isinstance(IC, np.ndarray):
            u.vector()[self.dofmap1.dofs()]  = IC
            u0.vector()[self.dofmap1.dofs()] = IC
        else:            
            u_init = IC(**(ICParameters | {'degree': 1}))
            u.interpolate(u_init)
            u0 = u.copy(deepcopy=True)
        
        self.c_init, self.mu_init, self.eta_init = u0.split(deepcopy=True)
        # Solve auxiliary problem to determine m0, eta0 (eta0 part is untested)
        if kwargs.get('autotune_mu0',False):
            u_V2 = Function(self.V1)
            u_V2.assign(self.c_init)
            # u_V2 =project(u1_sol,self.V2.collapse()) # This gives error on the order of machine precission
            # # DEBUG: Check if u_V2 and u1_sol coincide
            # print(max(u1_sol.vector()-u_V2.vector()))
            mu_V2  = Function(self.V2); 
            dmu_V2 = TrialFunction(self.V2) 
            v_V2   = TestFunction(self.V2)
            A0     = inner(dmu_V2,v_V2)*dx
            L0     = (self.phi1(u_V2) + self.phi2(u_V2))*v_V2*dx + lmbda*dot(grad(u_V2),grad(v_V2))*dx
            solve(assemble(A0), mu_V2.vector(), assemble(L0), "gmres", "ilu")
            # Test solution accuracy by calculating the residue
            m0_res = assemble((mu_V2*mu_V2 - (self.phi1(u_V2) + self.phi2(u_V2))*mu_V2-lmbda*dot(grad(u_V2),grad(mu_V2)))*dx)
            self.mu_init.assign(mu_V2)
            if self.verbose:
                print(f'Residue of calculated initial condition for mu is {m0_res}')
            self.eta_init.vector()[:] = 0

        ###-----------------------------------
        ### Weak statement of the equations
        ###-----------------------------------

        self.fg_LinearSolve = kwargs.get('LinSolve', False)

        Phi  = self.Phi
        phi1 = self.phi1
        phi2 = self.phi2
        M    = self.M
        Md   = self.Md
        dt   = self.dt
        if self.fg_LinearSolve:
            ###----------------------------
            ### Linear case assembly
            ###----------------------------

            self.set_LinSolver(**kwargs)

            v1, v2, v3  = TrialFunctions(W)
            w1, w2, w3  = TestFunctions(W)

            #Assemble the S_n that contains the gradient integral     
            #https://fenicsproject.org/qa/12634/gradient-of-a-cg-order-one-element/
            self.S0 = Vector(u0.vector())
            self.S0.zero()                                        # S0 is zero at the beginning
            self._S = dot(Md(c)*(c-c0)*grad(eta),grad(w1))*dx

            K = v1*w1*dx + dot(M(c0)*grad(v3),grad(w1))*dx - dot(Md(c0)*(c0-c1)*grad(v3),grad(w1))*dx\
              + v2*w2*dx - eval(sphi1.replace('x','v1'))*w2*dx - lmbda * dot(grad(v1), grad(w2))*dx \
              + v3*w3*dx - (b1+b2)*v2*w3*dx  
            self.StiffnessMatrix = assemble(K) 
            self._RHS = self.c_init*w1*dx + phi2(c0)*w2*dx 

            ### Initialize history term and modes (in numpy vector terms)
            v1m, w1m = TrialFunction(self.V1), TestFunction(self.V1)
            v2m, w2m = TrialFunction(self.V2), TestFunction(self.V2)
            v3m, w3m = TrialFunction(self.V3), TestFunction(self.V3)

            self.Modes      = np.zeros([len(self.dofmap3.dofs()), self.TS.nModes+1])
            self.History    = self.Modes @ self.TS.gamma_k

            # Solve the suplementary equations from the system for Modes
            self.Modes_EqS_Matrix    = assemble(v2m*w2m*dx)
            if   self.TS.Scheme[3:] == 'mIE':
                self.Modes_EqS_LF  = (phi1(c) + phi2(c0))*w2m*dx + lmbda * dot(grad(c), grad(w2m))*dx
            elif self.TS.Scheme[3:] == 'mCN':
                self.Modes_EqS_LF  = (phi1(c) + phi2(c))*w2m*dx + lmbda * dot(grad(c), grad(w2m))*dx
            else:
                raise Exception('Unknown scheme! Please set "scheme" parameter in the format "RA:mIE" or "RA:mCN".')
            self.mu    = Function(self.V2)
            self.mu.vector().zero()
            Modes_EqS_RHS= assemble(self.Modes_EqS_LF)
            self.Modes_LinSolver.solve(self.Modes_EqS_Matrix, self.mu.vector(), Modes_EqS_RHS)
            # self._CurrFlux = dot(M(c0)*grad(mu), grad(w1m))*dx
            # self._PrevFlux = dot(M(c0)*grad(mu0), grad(w1m))*dx
            self.Modes_Eq_muMatrix = assemble(v3m*w3m*dx)

        else:
            ###----------------------------
            ### General case assembly
            ###----------------------------

            ### Solver
            self.set_Solver(**kwargs)
            raise Exception('Nonlinear solution scheme is not implemented!')
        ###----------------------------------------------------------
        
        self.PrevSolFull = [u0, u1]
        self.CurrSolFull = u

        ### Other quantities
        self._mass     = c*dx

        # self._roughness = ((c - self.mass_init)**2)*dx

        # self._Energy1  = (Phi1(c)+Phi2(c))*dx + 0.5*lmbda*dot(grad(c), grad(c))*dx
        self._Energy   = Phi(c)*dx + 0.5*lmbda*dot(grad(c), grad(c))*dx
        # Print FEniCS parameters
        print(parameters.str(self.verbose))




        ###-----------------------------------------------------------

    def F(self, b, x):
        assemble(self._Residual, tensor=b)
        b[:] = b - self.History

    def J(self, A, x):
        assemble(self.Jacobian, tensor=A)

    def update_history_and_s(self):
        if self.scheme[:2] == 'RA': 
            if self.fg_LinearSolve:
                # In the case of parallel runtime the next code operatos on the locally-owned 
                # part of solution
                mu0 = self.mu.vector().get_local();
                Modes_EqS_RHS = assemble(self.Modes_EqS_LF)
                self.Modes_LinSolver.solve(self.Modes_EqS_Matrix, self.mu.vector(), Modes_EqS_RHS)
                mu = self.mu.vector().get_local();
                g_k, b1_k, b2_k = self.TS.gamma_k, self.TS.beta1_k, self.TS.beta2_k
                self.Modes[:]   = g_k*self.Modes + b1_k*np.reshape(self.Modes_Eq_muMatrix*mu,(-1,1)) + b2_k*np.reshape(self.Modes_Eq_muMatrix*mu0,(-1,1))
                self.History[:] = self.Modes @ g_k
                # Make sure that MPI vector is assembled properly
                # Only needed (presumably) for parallel runtime 
                # self.PrevSolFull[0].vector().apply("insert")                    
                # Calculate current value of S and save it as a previous for the next step
                self.S0 += assemble(self._S) 
            else:
                raise Exception('Nonlinear solution scheme is not implemented!')
        else:
            raise Exception('Unknown time-stepping scheme. Please use RA:XXX or L1:XXX, with XXX being "mIE", "mCN"')

    def __call__(self):
        u, u0, u1 = self.CurrSolFull, self.PrevSolFull[0], self.PrevSolFull[1]

        ### yield initial solution
        self.TS.restart()
        ## Save initial mass (needed for the calculation of roughness)
        self.mass_init = self.mass
        yield u.split(True)[0].vector()[:]

        # Step in time
        if self.fg_LinearSolve:
            while self.TS.inc():    
                # Shift the solution values one step back in time
                u1.vector().set_local(u0.vector().get_local())
                u0.vector().set_local(u.vector().get_local())
                # Update RHS of the system 
                RHS = assemble(self._RHS) + self.S0
                RHS[self.dofmap3.dofs()] += self.History
                # Solve
                self.LinSolver.solve(self.StiffnessMatrix, u.vector(), RHS)
                # Update history, modes and the cumulative itegral S0
                self.update_history_and_s()
                yield u.split(True)[0].vector()[:]
        else:
            while self.TS.inc():    
                # Shift the solution values one step back in time
                u1.vector().set_local(u0.vector().get_local())
                u0.vector().set_local(u.vector().get_local())
                self.Solver.solve(self, u.vector())
                self.update_history_and_s()
                yield u.split(True)[0].vector()[:]
    
    ### Solution iterator (time-integration)
    def SolutionIterator(self):
        return tqdm(enumerate(self()), total=self.TS.nTimeSteps)


    #----------------------------------------
    # Fractional Time Stepper
    #----------------------------------------
    def set_TimeStepper(self, **kwargs):
        scheme = kwargs.get('scheme', 'RA')[:2]
        if scheme == 'L1':
            self.TS = FracTS_L1(**kwargs)
        elif scheme == 'RA':
            self.TS = FracTS_RA(**kwargs)          
        elif scheme == 'GJ':
            self.TS = FracTS_GJ(**kwargs)
        else:
            raise Exception('Wrong timestepper scheme {0:s}'.format(scheme))
        self.scheme = scheme
        self.dt = self.TS.dt
        self.T  = self.TS.TimeInterval
        return self.TS

    #----------------------------------------
    # NL Solver
    #----------------------------------------
    def set_Solver(self, **kwargs):
        # Nonlinear solver
        # Print list of available FEniCS linear solvers and preconditioners 
        if self.verbose:
            list_linear_solver_methods()
            list_krylov_solver_preconditioners()
        ## Setup the nonlinear solver
        # For avilable list of solvers check 
        # https://bitbucket.org/fenics-project/dolfin/src/master/dolfin/nls/PETScSNESSolver.cpp 
        # proved to be working:
        # newtonls
        # newtontr
        self.Solver = PETScSNESSolver('newtontr')
        #self.Solver = NewtonSolver()
        self.Solver.parameters["linear_solver"] = "lu"
        #self.Solver.parameters["linear_solver"] = "tfqm"
        #self.Solver.parameters["linear_solver"] = "richardson"
        # self.Solver.parameters["linear_solver"] = "bicgstab"
        #self.Solver.parameters["linear_solver"] = "cg"
        #self.Solver.parameters["linear_solver"] = "gmres"
        #self.Solver.parameters["linear_solver"] = "umfpack"
        #self.Solver.parameters["linear_solver"] = "mumps"
        # self.Solver.parameters["preconditioner"] = "default"
        self.Solver.parameters["preconditioner"] = "ilu"
        #self.Solver.parameters["preconditioner"] = "hypre_euclid"
        #self.Solver.parameters["preconditioner"] = "sor"
        #self.Solver.parameters["preconditioner"] = "icc"
        #self.Solver.parameters["preconditioner"] = "hypre_amg"
        #self.Solver.parameters["preconditioner"] = "petsc_amg"
        #if self.Solver.parameters["linear_solver"] != "lu":
        #    self.Solver.parameters["preconditioner"] = "ilu"
        #self.Solver.parameters["convergence_criterion"] = "residual" #"incremental"
        #self.Solver.parameters["convergence_criterion"] ="incremental" #"residual" #"incremental"
        self.Solver.parameters["maximum_iterations"] = 1000
        self.Solver.parameters["relative_tolerance"] = 1e-6
        #self.Solver.parameters["krylov_solver"]["nonzero_initial_guess"] = True
        self.Solver.parameters["absolute_tolerance"] = 1.e-8
        # self.Solver.parameters["error_on_nonconvergence"] = True
        self.Solver.parameters["report"] = self.verbose

        ### Linear solver
        # LinSolver = PETScKrylovSolver("cg", "amg")
        LinSolver = PETScLUSolver("mumps")
        # LinSolver = PETScKrylovSolver("gmres", "ilu")
        # # LinSolver = PETScKrylovSolver("cg", "amg")
        # LinSolver.parameters["maximum_iterations"] = 1000
        # LinSolver.parameters["relative_tolerance"] = 1.e-10
        # LinSolver.parameters["absolute_tolerance"] = 1.e-8
        # LinSolver.parameters["error_on_nonconvergence"] = True
        # LinSolver.parameters["nonzero_initial_guess"] = True
        # LinSolver.parameters["monitor_convergence"] = False
        self.LinSolver  = LinSolver

        self.LinSolverMu = PETScLUSolver("mumps") #PETScKrylovSolver("cg", "icc")
        # self.LinSolverMu.parameters["maximum_iterations"] = 1000
        # self.LinSolverMu.parameters["relative_tolerance"] = 1.e-10
        # self.LinSolverMu.parameters["absolute_tolerance"] = 1.e-8
        # self.LinSolverMu.parameters["error_on_nonconvergence"] = True
        # self.LinSolverMu.parameters["nonzero_initial_guess"] = True
        # self.LinSolverMu.parameters["monitor_convergence"] = False


    #----------------------------------------
    # Linear Solver
    #----------------------------------------
    def set_LinSolver(self, **kwargs):

        ### Linear solver
        self.LinSolver = PETScLUSolver("mumps")
        # self.LinSolver = PETScKrylovSolver("gmres", "ilu")
        # # # LinSolver = PETScKrylovSolver("cg", "amg")
        # # LinSolver.parameters["maximum_iterations"] = 1000
        # self.LinSolver.parameters["relative_tolerance"] = 1.e-12
        # self.LinSolver.parameters["absolute_tolerance"] = 1.e-12
        # # LinSolver.parameters["error_on_nonconvergence"] = True
        # self.LinSolver.parameters["nonzero_initial_guess"] = True
        # # LinSolver.parameters["monitor_convergence"] = False

        # self.LinSolverM2 = PETScLUSolver("mumps") #PETScKrylovSolver("cg", "icc")
        # self.LinSolverMu.parameters["maximum_iterations"] = 1000
        # self.LinSolverMu.parameters["relative_tolerance"] = 1.e-10
        # self.LinSolverMu.parameters["absolute_tolerance"] = 1.e-8
        # self.LinSolverMu.parameters["error_on_nonconvergence"] = True
        # self.LinSolverMu.parameters["nonzero_initial_guess"] = True
        # self.LinSolverMu.parameters["monitor_convergence"] = False


        self.Modes_LinSolver = PETScLUSolver("mumps")
        # self.Modes_LinSolver = PETScKrylovSolver("cg", "icc")
        # self.Modes_LinSolver.parameters["relative_tolerance"] = 1.e-12
        # self.Modes_LinSolver.parameters["absolute_tolerance"] = 1.e-12
        # self.Modes_LinSolver.parameters["nonzero_initial_guess"] = True



        self.LinSolverD = PETScKrylovSolver("gmres", "petsc_amg")
        # self.LinSolverD = PETScKrylovSolver("cg", "icc")



    #----------------------------------------------------------------------
    # Prperties and postprocessing
    #----------------------------------------------------------------------

    @property
    def Energy(self):
        return assemble(self._Energy, **self._fenics_assembly_pars )


    @property
    def Residual(self):
        u = self.CurrSolFull.sub(0)
        mu = self.CurrSolFull.sub(1)
        lmbda = Constant(self.eps**2)
        b1, b2  = self.TS.beta1, self.TS.beta2
        P = self.phi1(u)*mu*dx + self.phi2(u)*mu*dx  + lmbda * dot(grad(u), grad(mu))*dx
        Eq1 = u*u*dx + (b1+b2)*dot(grad(self.M(u)*mu ), grad(u))*dx
        Eq2 = mu*mu*dx - P
        return assemble(Eq1+Eq2, **self._fenics_assembly_pars )
        # return assemble(self._Residual, **self._fenics_assembly_pars )
        # return self.CurrSolFull.split(True)[0].vector()[:]-self.mass_init)
        # return self.Energy + self.HistoryEnergy
    
    @property
    def mass(self):
        return assemble(self._mass, **self._fenics_assembly_pars )

    @property
    def roughness(self):
        return self.norm(self.CurrSolFull.split(True)[0].vector()[:]-self.mass_init)
        
    def norm(self, v):
        f = Function(self.W)
        f.vector()[self.dofmap1.dofs()] = v[:]
        sqr_norm_f = assemble(f.sub(0)*f.sub(0)*dx, **self._fenics_assembly_pars )
        norm_f = np.sqrt(sqr_norm_f)
        return norm_f


