import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
import os
import cv2 
from PIL import Image
import OneD.Global as GF

from scipy.interpolate import interp1d

###################################
# For HARMONIC OSCILATTOR
# r and f Parameters as written in notebook
def r(hbar,m,v,L):
    return hbar/(2*m*v*L)

def f(omega,v,L):
    return 0.5*L*omega/v
    
####################################

## FOR NON DIMENSIONALIZED, FIXED PHI SYSTEMS ONLY

def kick(psi,phi,r,dtau):
    a = -1j*dtau/r
    u = np.exp(a*phi)
    psi_new = np.multiply(u,psi)
    return psi_new

def drift(psi,r,dz,dtau):
    n = len(psi)

    psi_n = np.fft.fft(psi,n)
    k = 2*np.pi*np.fft.fftfreq(n,dz) #wave-number in fourier domain

    a = -1j*r*dtau
    u = np.exp(a*k**2)
    psi_new_n = np.multiply(u,psi_n)

    psi_new = np.fft.ifft(psi_new_n,n)
    return psi_new

###################################################################################
# FOR PHASE-SPACE DISTRIBUTION STUFF
###################################################################################
def Husimi_phase(chi,z,eta):
    N = len(chi)
    dz = z[1]-z[0]                                                  
        
    k = 2*np.pi*np.fft.fftfreq(N,dz)
    dk = k[1]-k[0]
    g = np.array([np.exp(-(z_0-z)**2 / (2*eta**2)) for z_0 in z])
    
    f = np.fft.fft(np.multiply(chi,g))
    f = np.fft.fftshift(f,axes=1)

    F = np.absolute(f)**2 
    F = np.transpose(F)

    #normalize it:
    Norm_const = np.sum(dz*dk*F)
    F = F/Norm_const
    return F

def Husimi_phase_V2(chi, z, eta):
    dz = z[1]-z[0]
    p = 2*np.pi*np.fft.fftfreq(len(chi),dz)
    p = np.fft.fftshift(p)
    z_kn, p_kn = np.meshgrid(z,p)
    
    F = Husimi_phase(chi,z,eta)
    return F, z_kn, p_kn