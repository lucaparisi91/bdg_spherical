from . import harmonicOscillatorIsotropic as ho
import numpy as np

class trappedGP:
    """
    Assumes to use use units of harmonic oscillator length
    """
    def __init__(self, g,mu):
        self.g=g
        self.mu=mu
    
    def buildBdGMatrix( self,x, psi0 ):
        """
        Inputs:
            x : radial positions, 1D array
            psi0: ground state wavefunction
        Outputs:
            Ground state wavefunction    
        """

        g=self.g
        mu=self.mu
        
        H0=ho.buildHamiltonian(x,spherical=True)
        B = np.zeros( np.array(H0.shape)*2 )
        B[:H0.shape[0],:H0.shape[1]]=H0
        B[ H0.shape[0]:,H0.shape[1]:]=-H0
        
        i1=np.arange(0,H0.shape[0])
        i2=i1 + H0.shape[0]

        B[i1,i1]+= 2*g*psi0**2 - mu
        B[i2,i2]-= 2*g*psi0**2 - mu
        B[i1,i2]+= g*psi0**2
        B[i2,i1]-= g*psi0**2
        return B
    