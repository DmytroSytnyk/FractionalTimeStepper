import numpy as np
# import random
from dolfin import *
# from tqdm import tqdm
# from copy import copy
# from scipy.special import gamma
import sys, os

# Class representing the intial conditions
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        # random.seed(2 + MPI.rank(MPI.comm_world))
        np.random.seed(0)
        self.nModes = 10
        self.corrlen = 0.2
        self.k0 = np.arange(self.nModes)
        self.k1 = np.arange(self.nModes)
        self.coefs  = np.random.normal(loc=0, scale=1/self.nModes**2, size=(self.nModes,self.nModes)) #* np.exp(-self.k0**2/(2*self.corrlen**2)) + self.k1**2))
        self.G0 = np.exp(-self.k0**2 * self.corrlen**2/2)
        self.G1 = np.exp(-self.k1**2 * self.corrlen**2/2)
        super().__init__(**kwargs)
    def eval(self, values, x):
        # ptb = 1/100
        # for i in range(len(x)):
        #     ptb *= cos(2 * pi * x[i])
        # values[0] =  1/50*(cos(2*pi*x[1])  + cos(2*pi*x[0])) #0.02*(0.5 - np.random.random())
        # values[0] = np.random.normal(loc=0.4, scale=0.01)
        # values[0] = 0.4 + 0.2*(0.5 - np.random.random())
        # values[0] = np.random.random()
        # values[0] =  0.4+1/50*(cos(1/4*pi*x[1])  + cos(4*pi*x[0]))
        # values[0] = np.random.normal(loc=0.5, scale=0.001)
        # values[0] = (np.cos(pi*self.k0*x[0])*self.G0) @ self.coefs @ (np.cos(pi*self.k1*x[1])*self.G1)
        #####################################################
        # IC enabled by default in Ustim's code
        #####################################################
        values[0] =  0.4 + 1/50 * cos(2*pi*x[0]) * cos(2*pi*x[1]) # * cos(2*pi*x[2])
        values[1] = 0.0; values[2] = 0.0
        #####################################################
        ## Purely random IC (taken from the Marnvin's code ##
        #####################################################
        #values[0] = 0.002*(0.5 - random.random()) #ICSmooth(x, 0.2501)
        # NOTE: By definition \mu_0 = \Psi'(\phi_0) - \varepsilon^2 \Delta \psi
        #values[1] = 0.0 
    def value_shape(self):
        return (3,)

class IC_TwoBubbles(UserExpression):
    def __init__(self, **kwargs):
        self.c = [ (0.5-0.2, 0.5), (0.5+0.2, 0.5) ]
        # self.c = [ (0.5-0.2, 0.5-0.2), (0.5+0.2, 0.5-0.2), (0.5+0.2, 0.5+0.2), (0.5-0.2, 0.5+0.2) ]
        self.r = 0.15
        self.eps = kwargs.pop('eps', 0.03)
        super().__init__(**kwargs)
    def eval(self, values, x):
        v = len(self.c)-1
        for c in self.c:
            r = sqrt( (x[0]-c[0])**2 + (x[1]-c[1])**2 )
            v += np.tanh((self.r-r)/(sqrt(2)*self.eps))
        values[0] = v
        values[1] = 0.0; values[2] = 0.0 
    def value_shape(self):
        return (3,)

class IC_FourBubbles(UserExpression):
    def __init__(self, **kwargs):
        d = 0.2
        self.c = [ (0.5-d, 0.5-d), (0.5+d, 0.5-d), (0.5+d, 0.5+d), (0.5-d, 0.5+d) ]
        self.r = 0.15
        self.eps = kwargs.pop('eps', 0.03)
        super().__init__(**kwargs)
    def eval(self, values, x):
        v = len(self.c)-1
        for c in self.c:
            r = sqrt( (x[0]-c[0])**2 + (x[1]-c[1])**2 )
            v += np.tanh((self.r-r)/(sqrt(2)*self.eps))
        values[0] = v
        values[1] = 0.0; values[2] = 0.0 
    def value_shape(self):
        return(3,)

class IC_BubbleSoup(UserExpression):
    def __init__(self, **kwargs):
        seed = kwargs.pop('seed',0)
        np.random.seed(seed)
        n  = kwargs.pop('n', 3)
        self.eps = kwargs.pop('eps', 0.03)
        R  = min(0.5, max(0.02,kwargs.pop('R', 0.2)))
        r  = max(0.02, kwargs.pop('r', 0.04))
        R = max(R,r); 
        r = min(R,r)
        # Generate rundom bubbles
        self.bR = r + np.random.rand(n)*(R-r)
        self.bC = np.random.rand(2, n)*(np.ones((2,n)) - 2*self.bR)+self.bR;
        self.bn = n;
        super().__init__(**kwargs)
    def eval(self, values, x):
        vr = self.bC - np.tile([[x[0]],[x[1]]],(1,self.bn))
        r = np.linalg.norm(vr,axis=0,ord=2)
        # The values is assumed to be between -1 and 1 so one has to divide by the number of bubbles
        values[0]  = sum(np.tanh((self.bR-r)/(sqrt(2)*self.eps)))+self.bn-1
        # Make sure that values is within [-1,1]
        # This is necessary if bubbles overlap
        values[0]  = max(-1,min(1,values[0])) 
        values[1] = 0.0; values[2] = 0.0 
    def value_shape(self):
        return(3,)

class IC_RandomField(UserExpression):
    def __init__(self, **kwargs):
        seed = kwargs.pop('seed', 0)
        self.eps  = kwargs.pop('eps', 0.002)
        self.c    = kwargs.pop('center', 0.5)
        np.random.seed(seed)
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = self.eps*(self.c - random.random()) #ICSmooth(x, 0.2501)
        values[1] = 0.0 
        values[2] = 0.0 
    def value_shape(self):
        return(3,)

class IC_LinearCombination(UserExpression):
    def __init__(self, **kwargs):
        IC_Names = ('IC_TwoBubbles', 'IC_FourBubbles', 'IC_BubbleSoup', 'IC_RandomField')
        n  = kwargs.pop('n', 1)
        self.ICs = []
        w = np.array(kwargs.pop('IC_weights', (1)))
        self.ICweights = w/w.sum()
        for i in range(1,n+1):
            _IC = kwargs.pop(f'IC{i}', '')
            if _IC in IC_Names:
                _IC_Pars = kwargs.pop(f'IC{i}_Parameters', '')
                self.ICs.append(eval(_IC+'(**' + str(_IC_Pars) + ')'))
            else:
                raise Exception(f'IC {_IC} is unknows. Use one of {IC_Names}')
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0; values[1] = 0; values[2] = 0
        v = values.copy()
        for i,IC in enumerate(self.ICs):
            IC.eval(v,x)
            values += v*self.ICweights[i]
    def value_shape(self):
        return(3,)
##===========================================================
#   Helpers
#===========================================================

class Exporter:

    def __init__(self, ProblemObj, **kwargs):
        self.ProblemObj = ProblemObj
        self.Folder     = kwargs.get('ExportFolder', None)
        self.FilePrefix = kwargs.get('FilePrefix', '')
        self.name       = kwargs.get('Name', 'sol')
        if self.Folder:
            os.makedirs(self.Folder, exist_ok=True)
            import pprint
            from utils import save_dict2json
            hash = save_dict2json(os.path.join(kwargs.get('ExportFolder', None),''),pprint.pformat(ProblemObj.config))
            if self.FilePrefix.strip() == "":
                self.FilePrefix = hash
                if ProblemObj.verbose:
                    print('Autogenerated export file prefix ("FilePrefix" parameter): ', hash) 
            self.vtk_file = File(os.path.join(self.Folder, self.FilePrefix +'sol.pvd'), "compressed")

    def test_export_every(self, export_every):
        if export_every is None:
            return True
        else:
            t = self.ProblemObj.TS.CurrTime
            h = self.ProblemObj.TS.dt
            tau  = export_every
            test = ( t%tau <= h/2 or tau-t%tau < h/2)
            return test

    def export_iv_vtk(self):
        phi0 =  self.ProblemObj.c_init
        phi0.rename(self.name+'0','At t=0')
        self.vtk_file << phi0

    def export_step_vtk(self, **kwargs):
        export_every = kwargs.get('export_every', None)
        if self.test_export_every(export_every):
            t = self.ProblemObj.TS.CurrTime
            u = self.ProblemObj.CurrSolFull
            u.rename(self.name,'')
            self.vtk_file << (u.split()[0], t)

    def export_step_npy(self, **kwargs):
        export_every = kwargs.get('export_every', None)
        field        = kwargs.get('field', self.ProblemObj.CurrSolFull.split()[0].vector()[:])
        if self.test_export_every(export_every):
            i = self.ProblemObj.TS.TimeStep
            np.save(os.path.join(self.Folder,prefix+name + '_i={0:d}'.format(i)), field)

    def export_npy(self, data, **kwargs):
        np.save(os.path.join(self.Folder,self.FilePrefix + self.name + '.npy'))

    def import_npy(self, **kwargs):
        return np.load(os.path.join(self.Folder,self.FilePrefix + self.name + '.npy'))

    def clear(self):
        os.system('rm ' + self.Folder + '*' )


