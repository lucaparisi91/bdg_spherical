import numpy as np
import scipy as sp
from scipy import special
import matplotlib.pylab as plt
import h5py

def generate_grid(left,right,shape):
    dx=(right-left)/shape
    x=left + (np.arange(0,shape)+0.5)*dx
    return (x)

def norm(psi0,r):
    return np.sum(psi0**2*4*np.pi*r**2)*(r[1]-r[0])
    

def solveBdG(B):
    """
    Solves all the eigen values using a dense matrix for a spherically symmetric hamiltonian
    Inputs
    ----------------------------------------------------------
    B : a BdG matrix built from an hamiltonian sherical model
    
    Outputs
    -----------------------------------------------------------
    e: energies of the excited states with strictly positive energy sorted
    v: 2D array, wavefunctions properly normalized
    """
    e,v=sp.linalg.eig(B,right=True)
    eta=1e-5
    indices_ordered=np.argsort(e)
    indices_ordered=indices_ordered[np.real(e[indices_ordered]) >eta]
    e=e[indices_ordered]
    v=v[:,indices_ordered]

    return e,v

def save(psi,filename):
    f=h5py.File(filename,"w")
    f.create_dataset("psi", data=psi)
    f.close()

def load(filename):
    f=h5py.File(filename)
    psi=np.array(f["psi"])
    f.close()
    return psi
