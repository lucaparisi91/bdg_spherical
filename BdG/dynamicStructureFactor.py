import numpy as np
import tqdm
import matplotlib.pylab as plt
from scipy import optimize
import pandas as pd
import seaborn as sns


def matrix_element_sq(q,psi0,u,v,r):
    """
    Returns the matrix element <n | \rho_q | 0 >. Does doe use an fft.
    q: momenta
    psi0: 1D array,ground state wavefunction
    u : 1D array, u(r) bogoliubov function
    v : 1D array, v(r) bogoliubov function
    """

    f=np.sum( 4*np.pi*(r[1]-r[0])*(u + v )*r*psi0*np.sin(q*r)/q )
    return np.sum(f)


def plot(data,q,omega,mu=None,show_recoil=True):
    """
    data : 2D array with dynamic structure values
    q: transferred momenta axis , 1D array
    omega : energy axis, 1D array
    show_recoil: plot \omega=q^2/2m over the heatmap
    mu: chemical potential, show the LDA estimate for a trapped GP
    """
    q_labels=["{:.2f}".format(_q) for _q in q]
    omega_labels=["{:.2f}".format(_omega) for _omega in omega]

    S_data=pd.DataFrame(data=data.transpose(),index=omega_labels,columns=q_labels)
    ax=sns.heatmap(S_data,xticklabels=len(q)//5,yticklabels=len(omega)//5)
    #ax=sns.heatmap(S_data)

    ax.axes.invert_yaxis()
    if show_recoil:
        sns.lineplot(x=q/np.max(q)*len(q), y=0.5*q**2/np.max(omega)*len(omega),color="r" ,label="recoil energy")
    #sns.lineplot(x=q/np.max(q)*len(q), y=0.5*q**2/np.max(omega)*len(omega) )
    Er=q**2/2
    if mu is not None:
            sns.lineplot(x=q/np.max(q)*len(q), y=(Er * np.sqrt(1+2*mu/Er) )/np.max(omega)*len(omega),color="g" ,
                         label=r"$E_r\sqrt{1+2\mu/E_r}$")

    plt.xlabel(r"$q$")
    plt.ylabel(r"$\omega$")


class dynamicStructureFactor:
    def __init__(self,e,vs,r,psi0):
        """
        e: energies
        vs: 2D array, vs[0:len(r),n]=u(r) ,vs[len(r):,n]=v(r)
        r: radial positions
        psi0: groundstate wavefunction 
        """
        self.e=e
        self.vs=vs
        self.r=r
        self.psi0=psi0

    def __call__( self,q ):
        """
        Inputs:
        q: wavevectors
        Outputs:
        S : 2D matrix of shape (len(q),len(e))
        """
        e=self.e
        vs=self.vs
        x=self.r

        S=np.zeros( (len(q),(len(e))))
        ks=np.arange(0,len(e))
        for k in tqdm.tqdm(ks):
            u=vs[0:len(x),k]
            v=vs[len(x):,k]
            S[:,k]=[np.abs(matrix_element_sq(_q,self.psi0,u,v,x))**2 for _q in q]
        return S



class dynamicStructureFactorLDA:

    def __init__(self,density,r,epsi,bins=10000):
        """
        density: density of the ground state
        r: spatial positions
        bins: number of divisions for the estimation of the d\epsi/dr
        epsi: function, takes as an argument momentum q in modulus and return the energy of the mode for an homogeneous system 
        """
        self.density=density
        self.epsi=epsi
        self.bins=bins
        self.r=r
    

    def __call__(self,q,omega):
        """
        q: momenta at which to evaluate the dynamic stricture factor
        omega: frequency at which to evaluate the dynamic structure factor
        """
        S=np.zeros( (len(q),len(omega)) )
        for iQ in range(len(q)):
            for iOmega in range(len(omega)):
                S[iQ,iOmega]=self._lda_point(q[iQ],omega[iOmega])
        return S


    def _lda_point( self, q , omega  ):
        """
        Evaluates the lda estimate for a scalar momentum q and a scalar energy  omega
        """
        epsi=self.epsi
        r=self.r
        density=self.density
        maxN=self.bins

        if omega <= epsi(q,0):
            return 0
        try:
            n=optimize.brentq(lambda n : epsi(q,n) - omega ,0,maxN)
        except ValueError:
            return 0
        if n > np.max(density):
            return 0
        i=np.argmin(np.abs(density-n))
        
        if i ==0 or i==len(r)-1:
            return 0
        
        dr=r[1] - r[0]
        depsidr=(epsi(q,density[i+1] ) - epsi(q,density[i-1 ] ))/(2*dr)
        depsidr=np.abs(depsidr)
        S=np.pi*4*r[i]**2*density[i]*q**2*0.5/omega*1/depsidr
        return( S)

