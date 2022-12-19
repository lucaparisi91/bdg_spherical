import numpy as np

def kineticEnergyMatrix(x,spherical=False):
    """
    Assumes u`(r)=0 at the boundaries 
    Inputs
    -----------------------
    x: radial coordinate, 1D array
    spherical: if true the laplacian has an additional $\frac{2}{r}\frac{d}{dr} $ term in the laplacian
    Outputs
    ------------------------
    M : dense kinetic energy matrix
    """

    dx=x[1]-x[0]
    M=np.zeros( (len(x),len(x)) )
    i=np.arange(0,len(x))
    shape=len(x)


    M[i,i]=1/(dx)**2
    j_u=i[0:shape-1] + 1
    i_u=i[0:shape-1]
    M[i_u,j_u]=-0.5/dx**2
    if spherical:
        M[i_u,j_u]+=-0.5/(x[i_u]*dx)

    i_d=i[1:]
    j_d=i_d - 1
    M[i_d,j_d]=-0.5/dx**2
    if spherical:
        M[i_d,j_d]+=0.5/(x[i_d]*dx)
    
    
    M[0,0]+=-0.5/dx**2 
    if spherical:
        M[0,0]+= 0.5/(x[0]*dx)

    M[-1,-1]+=-0.5/dx**2  
    if spherical:
        M[-1,-1]+=- 0.5/(x[-1]*dx)
    
    return(M)
