
from math import *
import numpy as np
from scipy.optimize import fsolve, minimize, fminbound, root
from scipy.signal import residue, residuez, tf2zpk, zpk2tf
from scipy.special import gamma, gammaincc, hyperu
import scipy
import scipy.integrate
import matplotlib.pyplot as plt
from time import time




#########################################################

class BasicRationalApproximation:

    def __init__(self, **kwargs):
        self.verbose = kwargs.get('verbose', False)        
        self.alpha   = kwargs.get('alpha', 0.5)
        assert(self.alpha>=0 and self.alpha<=1)
        
        self.d, self.c, self.c_inf = np.array([]), np.array([]), 0

        ### Functions
        self.func = np.vectorize(self._func)
        self.appx = np.vectorize(self._appx)
        self.err  = np.vectorize(self._err)
        self.func_ker = np.vectorize(self._func_ker)
        self.appx_ker = np.vectorize(self._appx_ker)
        self.err_ker  = np.vectorize(self._err_ker)
        self.err_ml  = np.vectorize(self._err_ml)


    def __call__(self, x):
        return self.appx(x)

    def _func(self, x):
        return x**(-self.alpha)

    def _appx(self, x):
        c, d, c_inf = self.c, self.d, self.c_inf
        return np.sum(c/(x+d)) + c_inf

    def _err(self, x):
        return np.abs(self._func(x) - self._appx(x))

    def _func_ker(self, x):
        return x**(self.alpha-1)/gamma(self.alpha)

    def _appx_ker(self, x):
        c, d = self.c, self.d
        return np.sum(c * np.exp(-d*x))
        ### No c_inf-term since the origin is not considered !

    def _err_ker(self, x):
        return np.abs(self.func_ker(x) - self.appx_ker(x))

    def _err_ml(self, x):
        nu = self.alpha
        c, d, c_inf = self.c, self.d, self.c_inf
        return np.abs( 1 - c_inf - np.sum( c * ml(-d*x, alpha=1, beta=1-nu))   )


    
    def plot(self, z):
        y_ref = self.func(z)
        y_apx = self.appx(z) 
        err   = self.err(z)

        if any(z.imag):
            z_im = z.imag
            x = z.real.min() + z.imag
            t = 1/x[::-1]

        y_ref_ker = self.func_ker(t)
        y_apx_ker = self.appx_ker(t)
        err_ker   = self.err_ker(t)
        
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(z_im,err)
        plt.title('Spectral error')

        plt.subplot(2,2,2)
        plt.plot(z_im, y_ref.real, 'b-',  label='Re[Ref]')
        plt.plot(z_im, y_apx.real, 'r--', label='Re[RA]' )
        plt.plot(z_im, y_ref.imag, 'k-',  label='Im[Ref]')
        plt.plot(z_im, y_apx.imag, 'm--', label='Im[RA]' )
        plt.legend()
        plt.title('Compare spectrum')

        plt.subplot(2,2,3)
        plt.plot(t,err_ker)
        plt.title('Kernel error')

        plt.subplot(2,2,4)
        plt.plot(t, y_ref_ker, 'b-',  label='Ref')
        plt.plot(t, y_apx_ker, 'r--', label='RA' )
        plt.legend()
        plt.title('Compare kernels')

        plt.show()

#==========================================================

class RationalApproximation_AAA(BasicRationalApproximation):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tol = kwargs.get('tol', 1.e-12)

        self.set_Support(**kwargs)
        self.exec_AAA()
        self.exec_PartialFractions()

        if self.verbose:
            print('alpha:   ', self.alpha)
            print('nModes:  ', self.nModes)
            print('c_infty: ', self.c_inf)
            print('c: ', self.c)
            print('d: ', self.d)
            print('tol: ', self.tol)
            print('nSuppPoints: ', self.nSupportPoints)
            if self.verbose == 'plot':
                self.plot()    


    #------------------------------
    
    def set_Support(self, **kwargs):
        fg_Reciprocal = kwargs.get('fg_Reciprocal', True)
        self.fg_Reciprocal = fg_Reciprocal

        ### TARGET FUNCTION !!!
        if fg_Reciprocal:
            self.target_func = lambda x: x**self.alpha
        else:
            self.target_func = lambda x: x**(-self.alpha)

        ### Support points
        if  'SupportPoints' in kwargs.keys():
            self.SupportPoints  = kwargs.get('SupportPoints')
            self.nSupportPoints = self.SupportPoints.size
            self.Zmin = 1/self.SupportPoints.max()
            self.Zmax = 1/self.SupportPoints.min()
        else:
            self.Zmin = kwargs.get('Zmin', 1)
            self.Zmax = kwargs.get('Zmax', 1.e3)
            assert(self.Zmin<self.Zmax)
            self.nSupportPoints = kwargs.get('nSupportPoints', 100)
            if fg_Reciprocal:
                self.SupportPoints  = np.geomspace(1/self.Zmax, 1/self.Zmin, self.nSupportPoints)
            else:
                self.SupportPoints  = np.geomspace(self.Zmax, self.Zmin, self.nSupportPoints)
            # self.SupportPoints  = 1/self.Zmax + (1/self.Zmin-1/self.Zmax)*np.linspace(0, 1, self.nSupportPoints)**5

        ### Maximal polynomial degree
        self.MaxDegree = kwargs.get('MaxDegree', 100)
        self.MaxDegree = np.min([self.MaxDegree, self.nSupportPoints-1])

    #------------------------------

    def exec_AAA(self):
        np.seterr(divide='ignore', invalid='ignore')    ### deal with divisions by zero

        M = self.nSupportPoints
        Z = self.SupportPoints.reshape([M,1])
        F = self.target_func(Z)
        F[np.isnan(F)] = 0

        SF = scipy.sparse.spdiags(F.flatten(),0,M,M)
        Wt = scipy.sparse.spdiags(np.ones_like(F).flatten(),0,M,M)
        # Wt = scipy.sparse.spdiags((1/F).flatten(),0,M,M)
        J = np.arange(M)
        J_opt = np.array([], dtype=np.int)
        z = np.array([]).reshape([0,1])
        f = np.array([]).reshape([0,1])
        C = np.array([]).reshape([M,0])
        errvec = np.array([])
        R = np.mean(F)*np.ones_like(F)
        for m in range(self.MaxDegree):
            j = np.argmax(np.abs(Wt @ (F-R)))
            J_opt = np.append(J_opt, j)
            z = np.vstack([z, Z[j]])
            f = np.vstack([f, F[j]])
            J = J[J!=j]
            C = np.hstack([ C, 1/(Z-Z[j]) ])
            Sf = np.diag(f.flatten())
            A = Wt @ (SF @ C - C @ Sf)
            U,S,V = scipy.linalg.svd(A[J,:])
            w = V[m,:].reshape([-1,1])
            N = C @ (w*f)
            D = C @ w
            R[:] = F
            R[J] = N[J]/D[J]
            err = np.linalg.norm(Wt @ (F-R), ord=inf)
            errvec = np.append(errvec, err)
            if self.verbose: print(err)
            if err <= self.tol: break
        if self.verbose: print('degree',m)
        m = w.size

        B = np.eye(m+1)
        B[0,0] = 0
        E = np.block([ [ 0, w.reshape([1,m]) ], [ np.ones([m,1]), np.diag(z.flatten()) ] ])
        pol = scipy.linalg.eig(E,B, left=False, right=False)
        pol = pol[~np.isinf(pol)]
        E = np.block([ [ 0, (w*f).reshape([1,m]) ], [ np.ones([m,1]), np.diag(z.flatten()) ] ])
        zer = scipy.linalg.eig(E,B, left=False, right=False)
        zer = zer[~np.isinf(zer)]

        ### Alternative method for zeros and poles
        # mask = np.arange(m)
        # P, Q = 0, 0
        # for k in range(w.size):
        #     mask_k = mask[mask!=k]
        #     rts = z[mask_k].flatten()
        #     p_k = np.poly(rts)
        #     P = P + w[k] * f[k] * p_k
        #     Q = Q + w[k] * p_k
        # zer = np.roots(P)
        # pol = np.roots(Q)

        ### Remove infinite roots
        pol = pol[np.abs(pol)<1.e13]
        zer = zer[np.abs(zer)<1.e13]

        ### Assertion: no imaginary part
        assert( np.all(np.isclose(np.imag(zer),0)) )
        assert( np.all(np.isclose(np.imag(pol),0)) )

        self.LeadingConst = np.sum(w*f) / np.sum(w)
        self.Zeros, self.Poles = zer.real, pol.real
        self.P, self.Q = np.poly(self.Zeros), np.poly(self.Poles)



    #------------------------------

    def exec_PartialFractions(self):
        zer, pol, a = self.Zeros, self.Poles, self.LeadingConst  

        ### Degenerated case (alpha=1)
        if self.alpha==1:
            self.c = np.array([1])
            self.d = np.array([0])
            self.c_inf = 0
            self.nModes = self.d.size
            return

        ### Degenerated case (alpha=0)
        if self.alpha==0:
            self.c = np.array([])
            self.d = np.array([])
            self.c_inf = 1
            self.nModes = 0
            return

        # P, Q = np.poly(zer), np.poly(pol)
        P, Q = zpk2tf(zer, pol, a)
    
        ### Invertion
        if self.fg_Reciprocal:
            P = P[::-1]
            Q = Q[::-1]
        if P[0]>0:
            a = P[0]/Q[0]
            P /= P[0]
            Q /= Q[0]
        else:
            a = P[1]/Q[0]
            P /= P[1]
            Q /= Q[0]        

        ### Partial fractions decomposition
        c, pol, res = residue(P, Q)#, tol=1.e-12)#self.tol)

        ### Assertions
        assert(res.size==1)
        assert(np.isclose(res[0].imag,0))
        assert(all(np.isclose(c.imag,0)))
        assert(all(np.isclose(pol.imag,0)))
        assert(np.isclose(np.imag(a),0))
        pol[np.isclose(pol, 0)] = 0

        self.d = -pol.real
        self.c = a*c.real
        self.c_inf = a*res[0].real
        self.nModes = self.d.size

        assert(np.all(self.d>=0))
        assert(np.all(self.c>=0))
        assert(self.c_inf>=0)
    

    #------------------------------
    # Postprocessing
    #------------------------------

    def plot(self, x=None):
        if x is None: x = self.Zmin + 1j*np.linspace(0, self.Zmax-self.Zmin, 10000)
        # if x is None: x = np.geomspace(self.Zmin, self.Zmax**(1/self.alpha), 10000)
        super().plot(x)

       
    def ra_bar(self, x, w, f, z):
        N, D = 0, 0
        for k in range(len(w)):
            if np.isclose(x,z[k]): return f[k]
            N = N + w[k]*f[k]/(x-z[k])
            D = D + w[k]/(x-z[k])
        return N/D

    def ra_mp(self, x, r, p, k0):
        y = 0 + k0
        for k in range(len(p)):
            y = y + r[k]/(x-p[k])
        return y

    #-----------------------------------------
    # Auxilary funtions for Error estimate
    #------------------------------ ----------

    def int_err(self, a=None, b=None):
        if a is None: a = 1/self.Zmax
        if b is None: b = 1/self.Zmin
        if self.c.size==self.d.size:
            return scipy.integrate.quad(self.err_ker, a, b, points=[0])[0]
        else:
            h = 1/self.Zmax
            nu = self.alpha
            I = scipy.integrate.quad(self.err_ker, h, 1/self.Zmin, points=[0],limit=1000)[0] + (h**nu/gamma(1+nu) - c[-1] - np.sum(self.c[:-1]/self.d*(1-np.exp(-self.d*h))))
            return np.abs(I)


    def int_err1(self):
        h = 1/self.Zmax
        nu = self.alpha
        c, d, c_inf = self.c, self.d, self.c_inf
        I = np.where(d>0, (1-np.exp(-d*h))/d, h)
        E1 = h**nu/gamma(1+nu) - c_inf - np.sum(c*I)
        E1 = np.abs(E1)
        return E1


    def int_err2(self):
        a = 100
        def func(z):
            return np.abs(self.func_ker(z) - self.appx_ker(z))
        I = scipy.integrate.quad(func, 1/self.Zmax, 1/self.Zmin, points=[0, 1/self.Zmax], limit=1000000)
        E2 = I[0]
        return E2

    def int_err1_ml(self):
        h = 1/self.Zmax
        nu = self.alpha
        c, d = self.c[:-1], self.d
        c_infty = self.c[-1]
        E1 = h**nu/gamma(1+nu) - c_infty - np.sum(c/d*(1-np.exp(-d*h)))
        E1 = np.abs(E1)
        return E1


    def int_err2_ml(self):
        def func(z):
            return self.func_ker(z) - self.appx_ker(z)
        I = scipy.integrate.quad(func, 1/self.Zmax, 1/self.Zmin, points=[0], limit=10000)
        E2 = np.abs(I[0])
        # print(I[1])
        return E2


    def err_tail(self):
        h = 1/self.Zmax
        nu = self.alpha
        c, d = self.c[:-1], self.d
        c_infty = self.c[-1]
        E1 = h**nu/gamma(1+nu) - c_infty - np.sum(c/d*(1-np.exp(-d*h)))
        E1 = np.abs(E1)
        return E2


    #------------------------------



######################################################################
#                           TESTING
######################################################################

if __name__ == "__main__":

    import sys, os
    from MittagLeffler import ml

    alpha = 0.9
    nu = alpha
    T  = 1
    dt = 1.e-5
    Zmin, Zmax = 1/T, 1/dt
    tol = 1.e-13
    nNodes = 100
    verbose = False

    RA = RationalApproximation_AAA( alpha=alpha,
                                    tol=tol, nSupportPoints=nNodes,
                                    Zmin= Zmin, Zmax= Zmax,
                                    verbose=verbose,
                                    fg_Reciprocal=False)
    c, d = RA.c, RA.d

    x = np.geomspace(1.e-12, 1/dt, 1000)

    plt.figure('Error')
    y_r = [ RA.err(z) for z in 1/T + x]
    y_c = [ RA.err(z) for z in 1/T + 1j*x]
    plt.plot(x, y_r, label='real')
    plt.plot(x, y_c,  '-', label='imag') 
    plt.hlines(dt,x.min(),x.max(),color='gray',linestyle='--')
    plt.hlines(dt**(1+nu),x.min(),x.max(),color='black',linestyle=':')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()

    x = np.geomspace(dt, 1.e-3, 10000)
    y_ex = np.array([ RA.func_ker(z) for z in x])
    y_ra = np.array([ RA.appx_ker(z) for z in x])


    plt.figure('Compare functions')    
    plt.plot(x,y_ex,'b-')
    plt.plot(x,y_ra,'r--')
    plt.legend(['Ref', 'RA'])
    plt.ylim([0,None])
    plt.xscale('log')


    plt.figure('Kernel error')
    plt.plot(x,np.abs(y_ex-y_ra),'b-')
    plt.xscale('log')

    E = np.trapz(np.abs(y_ex-y_ra), x=x)
    print(E)

    plt.show()