from . import  hamiltonian
import numpy as np

import scipy as sp
from scipy import special

def wavefunction(x,n,normalized=False,spherical=False):
    """
    inputs
    ------------------------------
    x: radial coordinates, 1D array
    n: index of the excited state
    """
    H=sp.special.hermite(n)
    C=1
    if not spherical:
        if normalized:
            C=1/( np.pi**4*np.sqrt(2**n*np.math.factorial(n)) )
            return H(x)*np.exp(-x**2*0.5)*C
        else:
            raise NotImplementedError("No spherical wavefunction yet.")


def buildHamiltonian(x,omega=1,spherical=True):
    """
    Inputs
    -----------------------------
    x: radial coordinates

    Outputs:
    ------------------
    H : dense hamiltonian matrix
    """

    dx=x[1]-x[0]
    H=hamiltonian.kineticEnergyMatrix(x,spherical=spherical)
    i=np.arange(0,len(x))
    H[i,i]+=0.5*omega**2*x**2
    return H

def densityTF(r,g,N):
    R=(15./(4*np.pi) * g * N)**(1/5)
    n0=0.5*R**2/g
    y=n0*(1 - (r/R)**2)
    y[y<0]=0
    return y