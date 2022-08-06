import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
import os
import cv2 
from PIL import Image
import OneD.Global as GF

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

def time_evolve_FixedPhi(psi,phi,r,dz,dtau,m):
    #1. Kick in potential
    psi_new = kick(psi,phi,r,dtau/2)

    #2. Drift in differential operator
    psi_new = drift(psi_new,r,dz,dtau)

    #3. Kick again (potential is not updated)
    psi_new = kick(psi_new,phi,r,dtau/2)

    #4. Calculate updated density perturbation
    rho_new = m*np.absolute(psi_new)**2
    P_new = rho_new-np.mean(rho_new)

    return psi_new,phi,P_new

def time_evolve(chi,phi,r,dz,dtau,m,L):
    #1. Kick in current potential
    chi_new = kick(chi,phi/2,r,dtau/2)

    #2. Drift in differential operator
    chi_new = drift(chi_new,r,dz,dtau)

    #3. Update potential
    phi_new = GF.fourier_potential(chi_new,L)

    #4. Kick in updated potential
    chi_new = kick(chi_new,phi_new/2,r,dtau/2)

    #5. Calculate updated density perturbation
    rho_new = np.absolute(chi_new)**2
    P_new = rho_new-np.mean(rho_new)

    return chi_new,phi_new,P_new

def time_evolveV2(chi,phi,r,dz,dtau,m,L):
    #1. Kick in current potential
    chi_new = kick(chi,phi/2,r,dtau/2)

    #2. Drift in differential operator
    chi_new = drift(chi_new,r,dz,dtau)

    #3. Update potential
    rho_new = np.absolute(chi_new)**2
    phi_new = GF.fourier_potentialV2(rho_new,L)

    #4. Kick in updated potential
    chi_new = kick(chi_new,phi_new/2,r,dtau/2)

    #5. Calculate updated density perturbation
    rho_new = np.absolute(chi_new)**2
    #P_new = rho_new-np.mean(rho_new)

    return chi_new,phi_new,rho_new

###################################################################################
# FOR PHASE-SPACE DISTRIBUTION STUFF
###################################################################################
def Husimi_phase(chi,z,dz,L,eta):
    N = len(chi)
    
    k = 2*np.pi*np.fft.fftfreq(len(z),dz)
    dk = k[1]-k[0]
    #print(f"k[N//2-1] = {k[N//2 -1]}, k[N//2] = {k[N//2]}")

    f_s = np.ndarray((N,N), dtype = complex)
    for i in range(len(z)):
        z_0 = z[i]

        g = np.exp(-(z_0-z)**2 / (2*eta**2))
        f = np.fft.ifft(np.multiply(chi,g))
        #f = np.multiply(np.exp(1j*k*x_0/2),f) #A*B*f

        
        f = np.append(f[N//2:N],f[0:N//2])
        
        f_s[i] = f

    F_s = np.absolute(f_s)**2 
    F_s = np.transpose(F_s)

    #normalize it:
    Norm_const = np.sum(dz*dk*F_s)
    F_s = F_s/Norm_const
   
    return F_s

def generate_phase_plots(psi_s,x,dx,L,m,hbar,max_F,eta,dt,frame_spacing, Directory,folder_name):
    
    k = 2*np.pi*np.fft.fftfreq(len(x),dx)
    k = k#/L #non-dimensionalize
    #rescale wavenumber k to velocity v:
    v = k*(hbar/m)

    x_min, x_max = np.min(x), np.max(x)
    v_min, v_max = np.min(v), np.max(v)
    
    #directory = "C:\\Users\\boris\\OneDrive - Queen's University\\Documents\\JOB_Files\\McDonald Institute Fellowship\\Research\\Coding"
    path = Directory+"/"+folder_name

    for i in range(0,len(psi_s),frame_spacing):
        F = Husimi_phase(psi_s[i],x,dx,L,eta)
        plt.figure(figsize = (10,10))
        plt.title(f"Time {round(dt*i,3)}")
        #plt.figure(figsize = (10,5))
        plt.imshow(F,extent = (x_min,x_max,v_min,v_max),cmap = cm.hot, norm = Normalize(0,max_F), aspect = (x_max-x_min)/(v_max-v_min))
        plt.xlim([x_min,x_max])
        plt.ylim([v_min,v_max])
        plt.colorbar()
        
        #now save it as a .jpg file:
        filename = 'ToyModelPlot' + str(i+1).zfill(4) + '.jpg'
        folder = path #"C:\\Users\\boris\\OneDrive - Queen's University\\Documents\\JOB_Files\\McDonald Institute Fellowship\\Research\\Coding\\"+folder_name
        plt.savefig(folder + "/" + filename)  #save this figure (includes both subplots)
        
        plt.close() #close plot so it doesn't overlap with the next one
        #print(i,' Done')
