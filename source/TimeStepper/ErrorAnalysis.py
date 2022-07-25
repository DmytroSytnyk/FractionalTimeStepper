
from cProfile import label
from errno import EROFS
import numpy as np
import matplotlib.pyplot as plt
from sympy import plot


class ErrorAnalysis:

    def __init__(self, Problem, **kwargs) -> None:
        self.Problem   = Problem
        

    #----------------------------------------------------------------------------------------
    #   Norms
    #----------------------------------------------------------------------------------------

    def normH(self, u):
        return self.Problem.Model.norm(u)

    def normT_weighted(self, u):
        t, alpha = self.time_grid, self.Problem.alpha
        return np.linalg.norm(u * t**(1-alpha), ord=2)

    def normT(self, u):
        return np.linalg.norm(u, ord=2) #*self.Problem.dt
        # return np.linalg.norm(u, ord=np.Inf)
        # return np.abs(u[-1])
        # n = len(u)//3
        # return np.linalg.norm(u[n:], ord=2)
        # return self.normT_weighted(u)


    #----------------------------------------------------------------------------------------
    #   Error computation
    #----------------------------------------------------------------------------------------

    def compute_local_error(self, **kwargs):
        fg_verbose = kwargs.get("verbose", False)
        fg_plot    = kwargs.get("plot", False)
        fg_show    = kwargs.get("show", True)

        ExactSolution  = list(self.Problem(ReferenceSolution=True))
        AppxSolution   = self.Problem()  ### Numerical solution: generaor, to get a list, call [AppxSolution]
        self.time_grid = np.arange(self.Problem.MaxTimeStep+1)*self.Problem.dt
        error = np.zeros_like(self.time_grid)
        sol_num = np.zeros_like(self.time_grid)
        sol_ref = np.zeros_like(self.time_grid)


        ### Time integration
        for i, u in enumerate(AppxSolution):
            u_ref    = ExactSolution[i]
            error[i] = self.normH(u-u_ref)#/self.normH(u_ref)   ### RELATIVE ERROR !!!
            sol_num[i] = self.normH(u)
            sol_ref[i] = self.normH(u_ref)
            if fg_verbose:
                print(f"Time step {i}: Error = {error[i]}")

        if fg_plot:
            plt.figure()

            plt.subplot(1,2,1)
            plt.plot(self.time_grid, error, "o-",)
            plt.yscale("log")
            plt.ylabel("error")
            plt.xlabel("time")

            plt.subplot(1,2,2)
            plt.plot(self.time_grid, sol_num, "-", label="numerical")
            plt.plot(self.time_grid, sol_ref, "--", label="reference")
            plt.ylabel("value")
            plt.xlabel("time")
            plt.legend()
            
            if fg_show: plt.show()

        return error, np.abs(sol_ref)


    def compute_global_error(self, **kwargs):
        loc_error, norm_ref = self.compute_local_error(**kwargs)
        error = self.normT(loc_error) / self.normT(norm_ref)
        return error


    def analize(self, update_case, **kwargs):
        result = []
        i = 0
        while update_case(self.Problem, i):
            error = self.compute_global_error()
            result.append(error)
            i += 1
        return result


    def convergence_analysis(self, **kwargs):
        ls_alphas   = kwargs.get("alphas", [0.1, 0.5, 0.9])
        ls_numsteps = kwargs.get("numsteps", [100, 200, 400, 800, 1600])
        h = 1/np.array(ls_numsteps)
        fg_show = kwargs.get("show", True)

        def update_case(pb, i):
            if i<len(ls_numsteps):
                pb.set_TimeSolver(nTimeSteps=ls_numsteps[i])
                pb.compute_Coefficients(alpha=alpha)
                return True
            else:
                return False

        plt.figure()

        for alpha in ls_alphas:
            self.Problem.set_alpha(alpha)
            error = self.analize(update_case=update_case)
            plt.plot(1/np.array(ls_numsteps), error, "o-", label=r"$\alpha={}$".format(alpha))
            plt.plot(h, error[0]*(h/h[0])**1, "--", label=r"$\Delta t^1$".format(alpha=alpha))
            plt.plot(h, error[0]*(h/h[0])**(1+alpha), "--", label=r"$\Delta t^{{1+{alpha}}}$".format(alpha=alpha))
            plt.plot(h, error[0]*(h/h[0])**2, "--", label=r"$\Delta t^2$")

        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("error")
        plt.xlabel(r"$\Delta t$")
        plt.legend()
        if fg_show: plt.show()


    def blending_coefficient_analysis(self, **kwargs):
        ls_alphas = kwargs.get("alphas", [0.1, 0.5, 0.9])
        ls_blending_coefficient = kwargs.get("gammas", np.linspace(0,1, 10))

        def update_case(pb, i):
            if i<len(ls_blending_coefficient):
                val = ls_blending_coefficient[i]
                pb.compute_Coefficients(blending_coefficient=val)
                return True
            else:
                return False

        plt.figure()

        for alpha in ls_alphas:
            self.Problem.set_alpha(alpha)
            error = self.analize(update_case=update_case)
            plt.plot(ls_blending_coefficient, error, "o-", label=r"$\alpha={}$".format(alpha))

        plt.yscale("log")
        plt.ylabel("error")
        plt.xlabel(r"$\gamma$")
        plt.legend()
        plt.show()

