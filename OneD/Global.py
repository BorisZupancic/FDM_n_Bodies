import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.image import AxesImage
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.special import erf as erf
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import os
import cv2 
from PIL import Image
import OneD.FDM as FDM
import OneD.NBody as NB

import multiprocessing as mp
import threading 

import psutil

from time import process_time

def checkMemory(mem_limit):
    memoryUsage = psutil.virtual_memory().percent
    overflow = False
    if memoryUsage > mem_limit:
        print(f"Memory usage exceeded budget of {mem_limit} percent.")
        overflow = True
    return overflow


#########################################################3
#FOURIER STUFF

# def fourier_gradient(phi,length):
#     n = len(phi)
#     L = length 

#     #1. FFT the density (perturbation)
#     phi_n = np.fft.fft(phi,n) #fft for real input
    
#     #2. Compute the fourier coefficients of phi
#     grad_n = np.array([]) #empty for storage. Will hold the fourier coefficients
#     for nn in range(len(phi_n)):
#         k = 2*np.pi*nn/L
#         val = -1j*phi_n[nn]*k
#         grad_n = np.append(grad_n,val)
    
#     #3. IFFT back to get Potential
#     grad = np.fft.ifft(grad_n,n) #use Phi_n as Fourier Coefficients
    
#     grad = np.real(grad)
#     return grad

# def gradient(f,length,type = 'Periodic'):
#     if type == 'Isolated':
#         N = len(f)
#         dx = length/(N-1)
#         grad = np.gradient(f,dx,edge_order = 2)
#         return grad
#     elif type == 'Periodic':
#         #grad = fourier_gradient(f,length)
#         #Have decided to do the same thing:
#         N = len(f)
#         dx = length/(N-1)
#         grad = np.gradient(f,dx,edge_order = 2)
        
#         return grad

def Periodic_Poisson(rho,length):
    """
    
    Routine for solving Poisson's Equation with periodic boundary conditions.
    
    """
    
    n = len(rho)
    L = length #length of box

    #1. FFT the norm-squared of the FDM-function (minus it's mean background)
    rho_avg = np.mean(rho)
    p = 4*np.pi*(rho-rho_avg)
    p_n = np.fft.fft(p,n) #fft for real input
    
    #2. Compute the fourier coefficients of phi
    phi_n = np.array([]) #empty for storage. Will hold the fourier coefficients
    for nn in range(len(p_n)):
        if nn == 0:
            val = 0 #the "Jean's Swindle"
        if nn >=1: #for the positive frequencies
            k = 2*np.pi*nn/L
            val = -p_n[nn]/k**2
        phi_n = np.append(phi_n,val)
    
    #3. IFFT back to get Potential
    phi = np.fft.ifft(phi_n,n) #use Phi_n as Fourier Coefficients
    phi = np.real(phi)
    #phi = phi - np.min(phi)

    return phi

def Isolated_Poisson(rho,L,G_tilde):
    """
    
    Routine for solving Poisson's Equation with isolated boundary conditions.
    
    """

    N = len(rho)
    dz = L/(N-1)

    #1. Extend the density / FDM-function
    rho = np.append(rho,np.zeros(len(rho)-1))
    
    #2. FFT the norm-squared of the FDM-function (minus it's mean background)
    p = 4*np.pi*rho
    p_n = np.fft.fft(p) #fft for real input
    
    #3. Compute the fourier coefficients of phi
    phi_n  = np.multiply(G_tilde,p_n)
    
    #4. IFFT back to get Potential
    phi = np.fft.ifft(phi_n) #use Phi_n as Fourier Coefficients
    phi = np.real(phi)
    phi = phi * (2*L/len(phi))

    #5. Truncate the potential back to original size of grid:
    phi = phi[len(phi)//2:]

    return phi

def fourier_potential(rho,length = None, type = 'Periodic', G_tilde = None):
    """
    Solve for the potential in Poisson's equation using an FFT method, under either Periodic or Isolated boundary conditions. 

    Has option for solving either Periodic or Isolated boundary conditions.

    Differs from `fourier_potential()` as it takes in density instead of a wavefunction.

    """
    
    if type == 'Periodic':
        phi = Periodic_Poisson(rho,length)
        return phi
    elif type == 'Isolated':
        phi = Isolated_Poisson(rho,length,G_tilde)
        return phi
    elif type == "James's":
        pass

def potential_solver(length, type = 'Periodic', G_tilde = None):
    if type == 'Periodic':
        solver = lambda rho : Periodic_Poisson(rho,length)
    elif type == 'Isolated':
        solver = lambda rho : Isolated_Poisson(rho,length,G_tilde)
    return solver

#####################################################
# DIAGNOSTICS / TRACKERS
def stars_tracker(track_stars : bool, variable_mass, z):
    dz = z[1]-z[0]

    if track_stars == True:
        if variable_mass[0]==True:
            def stars_KW(stars, rho_part1, rho_part2, phi):
                K_array=[stars[0].get_K(),stars[1].get_K()]
                W_array=[stars[0].get_W(z,phi),stars[1].get_W(z,phi)]

                K = np.array([np.sum(K_array[0]),np.sum(K_array[1])])
                W = 0.5*np.array([np.sum(W_array[0]),np.sum(W_array[1])])

                W_2 = np.array([dz * np.sum(rho_part1*z*np.gradient(phi,dz)),dz * np.sum(rho_part2*z*np.gradient(phi,dz))])
            
                return K, W, W_2
        else: 
            def stars_KW(stars, rho_part1, rho_part2, phi):
                K_array=stars.get_K()
                W_array=stars.get_W(z,phi)

                K = np.sum(K_array)
                W = 0.5*np.sum(W_array)

                W_2 = dz * np.sum(rho_part1*z*np.gradient(phi,dz))
            
                return K, W, W_2
    else:
        def stars_KW(stars, rho_part1, rho_part2, phi):
            return None, None, None
    
    return stars_KW  

def stars_RMS_tracker(track_stars_rms : bool, variable_mass):
    if track_stars_rms == True:
        if variable_mass[0]==True:
            def stars_rms(stars):
                z_rms = np.sqrt(np.mean([*(stars[0].x-np.mean(stars[0].x))**2,*(stars[1].x-np.mean(stars[1].x))**2]))
                v_rms = np.sqrt(np.mean([*(stars[0].v-np.mean(stars[0].v))**2,*(stars[1].v-np.mean(stars[1].v))**2]))
                return z_rms, v_rms
        else:
            def stars_rms(stars):
                z_rms = np.sqrt(np.mean((stars.x-np.mean(stars.x))**2))
                v_rms = np.sqrt(np.mean((stars.v-np.mean(stars.v))**2))
                return z_rms, v_rms

        def get_rms(stars, z_rms_storage, v_rms_storage):
            z_rms_storage.append(stars_rms(stars)[0])
            v_rms_storage.append(stars_rms(stars)[1]),
            return z_rms_storage, v_rms_storage
    else: 
        def get_rms(stars, z_rms_storage, v_rms_storage):
            return z_rms_storage, v_rms_storage

    return get_rms

def FDM_tracker(track_FDM : bool, z, r):
    dz = z[1]-z[0]
    L = z[-1]-z[0]
    N = len(z)
    k = 2*np.pi*np.fft.fftfreq(N,dz)
    dk = k[1]-k[0]

    if track_FDM == True:
        def FDM_KW(chi,rho_FDM,phi):
            W = dz*np.sum(rho_FDM*z*np.gradient(phi,dz))
            
            chi_tilde = np.fft.fft(chi,norm = "ortho")            
            K = r * dz * np.sum(k**2 * np.abs(chi_tilde)**2)

            return K, W
    else:
        def FDM_KW(chi,rho_FDM,phi):
            return None, None

    return FDM_KW
    
#####################################################
#Full Calculation/Simulation Functions

def main_plot(type,G_tilde,L,eta,
                z,dz,mu,chi,rho_FDM,rho_part1,rho_part2,
                stars,Num_bosons,Num_stars,dtau,i,
                x_min,x_max,v_min,v_max,
                y00_max,y10_max,y01_max,y11_max,
                a_max,
                Directory = None,folder_name = None,track_stars = None, track_centroid = False, variable_mass=[False]):

    
    rho_part = rho_part1 + rho_part2
    # rho_part1, rho_part2 = NB.particle_density(stars,L,z,variable_mass)
    phi_part1 = fourier_potential(rho_part1,L,type = type, G_tilde = G_tilde)
    phi_part2 = fourier_potential(rho_part2,L,type = type, G_tilde = G_tilde)
    
    phi_part1 -= np.min(phi_part1)
    phi_part2 -= np.min(phi_part2) 

    layout = [['upper left', 'upper right', 'far right'],
                        ['lower left', 'lower right', 'far right']]

    fig, ax = plt.subplot_mosaic(layout, sharex = True, constrained_layout = True)
    fig.set_size_inches(20,10)
    plt.suptitle("Time $\\tau = $" +f"{round(dtau*i,5):.5f}".zfill(5), fontsize = 20)    
    ax['upper left'].set_title("Densities and Potentials",fontsize = 15)
    ax['upper right'].set_title("Phase Space Distributions", fontsize = 15)
    ax['far right'].set_title("Force contributions",fontsize = 15)
    
    ax['far right'].set_xlabel("$z = x/L_s$")
    ax['far right'].set_ylabel("Acceleration Field (code units)")
    ax['upper left'].set_xlabel("$z$")
    ax['lower left'].set_xlabel("$z$")
    ax['upper right'].set_xlabel("$z$")
    ax['lower right'].set_xlabel("$z$")
    ax['upper right'].set_ylabel("Velocity (code units)")
    ax['lower right'].set_ylabel("Velocity (code units)")

    ##############################################
    ax['far right'].set_ylim(-a_max/2,a_max/2)
    if variable_mass[0] == False:
        Part_force = -np.gradient(fourier_potential(rho_part,L,type = type, G_tilde = G_tilde),dz)
        ax['far right'].plot(z, Part_force, label = "Particle Contribution")

        ax['lower left'].plot(z, phi_part1, label = "$\\phi_{Quasi}$")
        ax['lower left'].plot(z, rho_part1, label = "$\\rho_{Quasi}$")
    else:
        Part1_force = -np.gradient(fourier_potential(rho_part1,L,type = type, G_tilde = G_tilde),dz)
        Part2_force = -np.gradient(fourier_potential(rho_part2,L,type = type, G_tilde = G_tilde),dz)
        ax['far right'].plot(z, Part1_force, label = "(Quasi) Particle Contribution")
        ax['far right'].plot(z, Part2_force, label = "(Light) Particle Contribution")
        ax['far right'].set_ylim(-a_max/2,a_max/2)
        
        ax['upper left'].plot(z, phi_part1, label = "$\\phi_{Quasi}$")
        ax['upper left'].plot(z, rho_part1, label = "$\\rho_{Quasi}$")
        
        ax['lower left'].plot(z,phi_part2,label = "$\\Phi_{Particles}$")
        ax['lower left'].plot(z,rho_part2,label = "$\\rho_{Particles}$")

        ax['upper right'].set_ylim(-y11_max,y11_max)
    
    # FDM
    if Num_bosons != 0:
        FDM_force = -np.gradient(fourier_potential(rho_FDM,L,type = type, G_tilde = G_tilde),dz)
        ax['far right'].plot(z, FDM_force, label = "FDM Contribution")
        
    
        phi_FDM = fourier_potential(rho_FDM,L,type = type, G_tilde = G_tilde)
        phi_FDM -= np.min(phi_FDM)
        ax['upper left'].plot(z,phi_FDM,label = "$\\Phi_{FDM}$")
        ax['upper left'].plot(z,rho_FDM,label = "$\\rho_{FDM} = m_{FDM}|\\psi|^2$")
        
    
    
    
    if Num_bosons !=0:
        #PHASE SPACE CALCULATION:
        F, z_kn, p_kn = FDM.Husimi_phase_V2(chi, z, eta) 
        
        
        ax['upper right'].pcolormesh(z_kn, p_kn/mu, F)
        
        if v_max<np.max(p_kn/mu):
            ax['upper right'].set_ylim(-v_max,v_max)
        else:    
            ax['upper right'].set_ylim(np.min(p_kn)/mu,np.max(p_kn)/mu)
        #ax['upper right'].colorbar()
        #divider = make_axes_locatable(ax["upper right"])
        #cax = divider.new_horizontal(size = '5%',pad = 0.05, pack_start = True)
        #fig.add_axes(cax)
        #fig.colorbar(mappable = im, cax = cax, ax=ax["upper right"],shrink = 0.75)
    
    ##############################################3
    # PARTICLES
    
    #Plot the Phase Space distribution
    if variable_mass[0] == True:
        #Plot light particles
        x1_s = stars[1].x
        v1_s = stars[1].v 
        ax['lower right'].scatter(x1_s,v1_s,s = 0.01,c = stars[1].mass, alpha=0.9, label = "(Light) Particles")
        #Plot heavy particles 
        x2_s = stars[0].x
        v2_s = stars[0].v 
        ax['upper right'].scatter(x2_s,v2_s,s = 8,c = stars[0].mass, label = "(Quasi) Particles")
    else:
        x_s = stars.x
        v_s = stars.v
        ax['lower right'].scatter(x_s,v_s,c = stars.mass, s = 0.01,alpha=0.9,label = "Particles")
        
        
    #ADDITIONAL:
    # Plotting the paths of those select stars
    # if track_stars == True and Num_stars >= 5:
    #     #E_array = np.array([])
    #     for j in range(5):
    #         ax['lower right'].scatter(stars[j].x, stars[j].v, c = 'k', s = 50, marker = 'o')
    #     #    E_array = np.append(E_array,stars[j].v**2)
    #     #E_storage = np.append(E_storage,[E_array])
        
    #PLOT CENTROID IN PHASE SPACE
    if Num_stars != 0:#only calculate if there are stars
        if variable_mass[0]==True:
            part1_centroid_z = [np.mean(stars[0].x)]
            part1_centroid_v = [np.mean(stars[0].v)]
            part2_centroid_z = [np.mean(stars[1].x)]
            part2_centroid_v = [np.mean(stars[1].v)]
            part_centroid_z = np.mean([part1_centroid_z,part2_centroid_z])
            part_centroid_v = np.mean([part1_centroid_v,part2_centroid_v])
            
            ax['upper right'].plot([-L/2,L/2],[part1_centroid_v,part1_centroid_v],"k--")
            ax['upper right'].plot([part1_centroid_z,part1_centroid_z],[-y11_max,y11_max],"k--")

            ax['lower right'].plot([-L/2,L/2],[part2_centroid_v,part2_centroid_v],"k--")
            ax['lower right'].plot([part2_centroid_z,part2_centroid_z],[-y11_max,y11_max],"k--")
        else:
            part_centroid_z = np.sum(stars.mass*stars.x)/np.sum(stars.mass)
            part_centroid_v = np.sum(stars.mass*stars.v)/np.sum(stars.mass)
        
            #ax['lower right'].scatter(part_centroid_z,part_centroid_v,s = 100,c = "r",marker = "o")
            ax['lower right'].plot([-L/2,L/2],[part_centroid_v,part_centroid_v],"k--")
            ax['lower right'].plot([part_centroid_z,part_centroid_z],[-y11_max,y11_max],"k--")
        part_centroid = np.array([part_centroid_z,part_centroid_v])
    else:
        part_centroid = None

    if Num_bosons !=0:
        fdm_centroid_z = np.sum(np.conj(chi)*z*chi) / np.sum(np.abs(chi)**2)
        k_mean = np.sum(np.conj(chi)*(-1j)*np.gradient(chi,dz))
        # mu = np.mean(rho_FDM/(np.absolute(chi)**2))
        fdm_centroid_v = k_mean/mu

        fdm_centroid = np.array([np.real(fdm_centroid_z),np.real(fdm_centroid_v)])
    else:
        fdm_centroid = None
    
    if ax['upper left'].lines:
        ax['upper left'].legend(fontsize = 15)
    if ax['upper right'].lines or ax['upper right'].collections: #or bool(ax['upper right'].get_images()):
        ax['upper right'].legend(fontsize = 15)
    if ax['lower left'].lines:
        ax['lower left'].legend(fontsize = 15)
    if ax['lower right'].collections:
        ax['lower right'].legend(fontsize = 15) 
    
    ax['far right'].legend(fontsize = 20)

    ax['lower left'].set_ylim(-y10_max,y10_max)
    ax['upper left'].set_ylim([-y00_max, y00_max] )
    ax['lower right'].set_ylim(-y11_max,y11_max)
    ax['lower right'].set_xlim(-L/2,L/2)
    ax['far right'].set_ylim(-a_max,a_max)
    if Num_bosons!=0 and Num_stars!=0:
        ymax = np.min([y11_max,np.max(p_kn)/mu])
        ymin = np.max([np.min(p_kn)/mu,-y11_max])
        ax['upper right'].set_ylim(ymin,ymax)

    #now save it as a .jpg file:
    folder = Directory + "/" + folder_name
    filename = 'Plot' + str(i).zfill(4) + '.jpg';
    plt.savefig(folder + "/" + filename)  #save this figure (includes both subplots)
    plt.clf()
    plt.cla()
    plt.close(fig) #close plot so it doesn't overlap with the next one

    if track_centroid == True:
        return part_centroid, fdm_centroid
    else:
        return None, None

def run_FDM_n_Bodies(sim_choice2, dtau, dynamical_times, t_dynamical, bc_choice, z, 
                    mu, Num_bosons, r, chi, 
                    stars, 
                    v_s, L_s, zmax, vmax, 
                    Directory, folder_name, 
                    absolute_PLOT = True, track_stars = False, track_stars_rms = False, track_centroid = False, fixed_phi = False,
                    track_FDM = False, variable_mass = False, history = False):
    
    st = process_time()
    #########################################
    #RETRIEVE INFO FROM INITIAL STARTUP/CONDITIONS
    L = z[-1]-z[0]
    dz = z[1]-z[0]
    N = len(z)

    #PHASE SPACE STUFF
    eta=(z[-1]-z[0])/np.sqrt(np.pi*len(chi)/2)
    k = 2*np.pi*np.fft.fftfreq(len(chi),dz)
    dk = k[1]-k[0]      
    #rescale wavenumber k to velocity v:
    hbar = 1
    v = k*(hbar/mu)
    x_min, x_max = np.min(z), np.max(z)
    v_min, v_max = np.min(v), np.max(v)
    if v_max >= np.abs(v_min):
        v_min = -v_max 
    elif np.abs(v_min) > v_max:
        v_max = -v_min
    print(f"v_min = {v_min}, v_max = {v_max}")
    if variable_mass[0] == True:
        Num_stars = len(stars[0].x) + len(stars[1].x)
    else:
        Num_stars = len(stars.x)


    #######################################################
    # DEFINE ROUTINES (to avoid redundant if-statements in main loop)
    #1. Choose Poisson-Solving Routine:
    if bc_choice == 1:
        type = 'Isolated'
        z_long = np.linspace(-L,L,2*N-1)
        G = 0.5*np.abs(z_long)
        G_tilde = np.fft.fft(G)    
    elif bc_choice == 2:
        type = 'Periodic'
        G_tilde = None
    get_phi = lambda rho : fourier_potential(rho,L,type=type,G_tilde=G_tilde)
    #Now option to freeze the potential:
    if fixed_phi == True:
        if variable_mass[0]==True:
            sigma = stars[0].mass
        else:
            sigma = stars.mass
        phi = 0.5*sigma*((z**2) / (L/2)**2 - 1)  #equal to 0 at boundaries 
        get_phi = lambda rho : phi 

    #2. Choose Density-getting Routine:
    #For FDM:
    if Num_bosons!=0:
        get_rho_FDM = lambda chi : mu*np.absolute(chi)**2
    else: 
        get_rho_FDM = lambda chi : np.zeros_like(z)
    #For Stars
    if Num_stars !=0:
        if variable_mass[0]==True:
            #smooth/filter the QP component:
            # from scipy.ndimage import gaussian_filter
            get_rho_part1 = lambda stars : NB.particle_density(stars[0],L,z)
            # get_rho_part1 = lambda stars : gaussian_filter(NB.particle_density(stars[0],L,z), sigma = 1) #L/len(stars[0].mass))
            # from scipy.signal import convolve, get_window
            # std = 0.5*N/len(stars[0].mass) #0.5*N/len(stars[0].mass)
            # window = get_window(("gaussian", std), N, fftbins=False)
            # window /= np.sum(window) #np.sqrt(2*np.pi*std**2)
            # print(f"std = {std}")
            # print(f"np.sum(window)={np.sum(window)}")
            # get_rho_part1 = lambda stars : convolve(NB.particle_density(stars[0],L,z),window, mode='same')
            get_rho_part2 = lambda stars : NB.particle_density(stars[1],L,z)
        else:
            # from scipy.signal import convolve, get_window
            # std = 0.5*N/len(stars.mass)
            # window = get_window(("gaussian", std), N, fftbins=False)
            # window /= np.sum(window) #np.sqrt(2*np.pi*std**2)
            # print(f"std = {std}")
            # print(f"np.sum(window)={np.sum(window)}")
            # get_rho_part1 = lambda stars : convolve(NB.particle_density(stars,L,z),window, mode='same')
            get_rho_part1 = lambda stars : NB.particle_density(stars,L,z)
            get_rho_part2 = lambda stars : np.zeros_like(z)
    else:
        get_rho_part1 = lambda stars : np.zeros_like(z)
        get_rho_part2 = lambda stars : np.zeros_like(z)

    get_rho_part = lambda stars : get_rho_part1(stars) + get_rho_part2(stars)
    get_rho = lambda chi,stars : get_rho_FDM(chi) + get_rho_part(stars) 

    #3. Kick and Drift Routines for Stars
    if variable_mass[0] == True:
        def kick(stars,g_interp,dtau):
            g_0 = g_interp(stars[0].x)
            stars[0].kick(g_0,dtau/2)
            g_1 = g_interp(stars[1].x)
            stars[1].kick(g_1,dtau/2)
            return stars 
        def drift(stars,dtau):
            stars[0].drift(dtau)
            stars[1].drift(dtau)
            return stars
    else:
        def kick(stars,g_interp,dtau):
            g_s = g_interp(stars.x)
            stars.kick(g_s,dtau/2)
            return stars 
        def drift(stars,dtau):
            stars.drift(dtau)
            return stars
        
    ##############################################
    # INITIAL SETUP
    rho_FDM = get_rho_FDM(chi)
    rho_part1 = get_rho_part1(stars)
    print(f"QP total mass: {np.sum(rho_part1)*dz}")
    rho_part2 = get_rho_part2(stars)
    rho_part = rho_part1+rho_part2

    #Check how it's normalized:
    print(f"integral of |chi|^2 : {np.sum(dz*rho_FDM)}")
    
    rho = get_rho(chi,stars)
    #check normalization
    print(f"Density Normalization Check: {np.sum(rho)*dz}")

    Part_force = -np.gradient(fourier_potential(rho_part,L, type = type, G_tilde = G_tilde),dz)
    FDM_force = -np.gradient(fourier_potential(rho_FDM,L, type = type, G_tilde = G_tilde),dz)

    a1 = np.abs([np.max(Part_force),np.min(Part_force)])
    a2 = np.abs([np.max(FDM_force),np.min(FDM_force)])

    a_max = np.max(np.append(a1,a2))*2
    print(f"a_max = {a_max}")   

    ##########################################################
    #PLOT AXIS LIMITS:
    #y0_max = np.max(phi)*1.5
    y00_max = np.max([np.max(rho_FDM),np.max(rho_part1)])*3
    # y10_max = np.max(rho_part)*3

    # if Num_bosons !=0:
    #     y00_max = np.max(rho_FDM)*3
    if Num_stars !=0:
        if variable_mass[0]==True:
            y00_max = np.max(rho_part1)*2
            y10_max = np.max(rho_part2)*3
        else:
            y10_max = np.max(rho_part)*3

    

    if Num_stars == 0:
        y10_max = y00_max
    if Num_bosons == 0 and variable_mass[0]!=True:
        y00_max = y10_max
    
    if Num_stars !=0:
        if variable_mass[0] == True:
            y01_max = 2*np.max([*stars[0].v,*stars[1].v])
        else:
            y01_max = 2*np.max(stars.v)
    else:
        v_max = 3*vmax
        pmax = vmax*mu
        p = np.linspace(-pmax,pmax,N)
        dp = p[1]-p[0]
        y01_max = vmax
        print(f"y01_max = {y01_max}")
    y11_max = y01_max
    
    ####################################################
    #PRE-LOOP TIME-SCALE SETUP
    collapse_index = int(np.ceil(t_dynamical/dtau))
    
    dtau = (collapse_index**(-1))*t_dynamical
    print(f"dtau = {dtau}")
    tau_stop = dynamical_times*t_dynamical 
    i_stop = dynamical_times * collapse_index
    if sim_choice2 == 1:
        dtau = 0.1*t_dynamical/5
        collapse_index = int(np.ceil(t_dynamical/dtau))
        i_stop = dynamical_times * collapse_index
        
        snapshot_indices = None 
            
    elif sim_choice2 == 2:
        # indices = [0,4*10,4*50,4*100,4*500]
        indices=[0,4*10,4*50,4*100,4*500]
        # x = 0 
        # while 2**x <= dynamical_times:
        #     indices.append(2**x)
        #     x+=1

        snapshot_indices = np.multiply(collapse_index,indices)

        if dynamical_times*collapse_index not in snapshot_indices:
            snapshot_indices = np.append(snapshot_indices, dynamical_times*collapse_index)

    
        print(f"Snapshots at i = {snapshot_indices}")
    
    

    print(f"Sim will stop at tau = {tau_stop}")
    time = 0
    # i = 0 #counter, for saving images
    
    if absolute_PLOT == True:
        os.chdir(Directory + "/" + folder_name) #Change Directory to where Image Folders are
    

    ######################################
    # DIAGNOSTICS + STORAGE SETTUP
    acceleration_storage = []

    if track_stars == True or track_FDM == True:
        if sim_choice2 == 1:
            snapshot_indices = np.multiply(collapse_index,indices)        
            track_snapshot_indices = snapshot_indices
        elif sim_choice2 == 2:
            track_snapshot_indices = snapshot_indices

    K_star_fine_storage = np.ndarray(i_stop+1) #Total Energy at each timestep
    W_star_fine_storage = np.ndarray(i_stop+1) 
    W_2_star_storage = np.ndarray(i_stop+1) 
    if variable_mass[0]==True:
        K_star_fine_storage = np.ndarray((i_stop+1,2)) #Total Energy at each timestep
        W_star_fine_storage = np.ndarray((i_stop+1,2))
        W_2_star_storage = np.ndarray((i_stop+1,2))
        
    get_stars_KW = stars_tracker(track_stars, variable_mass, z)

    if variable_mass[0]==True:
        K_star_storage = np.array([[np.zeros(len(stars[0].x)),np.zeros(len(stars[1].x))] for i in range(2)], dtype=object) #Total Energy at each timestep
        W_star_storage = np.array([[np.zeros(len(stars[0].x)),np.zeros(len(stars[1].x))] for i in range(2)], dtype=object)
    else:
        K_star_storage = np.ndarray((2,Num_stars)) #Kinetic of each star, at start and end
        W_star_storage = np.ndarray((2,Num_stars)) #Potential of each star, at start and end

    z_rms_storage = [] #None
    v_rms_storage = [] #None
    get_stars_rms = stars_RMS_tracker(track_stars_rms, variable_mass)

    R_half = []

    #FDM DIAGNOSTICS 
    W_FDM_storage = np.ndarray(i_stop+1) #None
    K_FDM_storage = np.ndarray(i_stop+1) #None    
    get_FDM_KW = FDM_tracker(track_FDM,z,r)

    et = process_time()
    print(f" `Startup` CPU Time = {et-st}")
    #################################################################
    # MAIN LOOP #####################################################
    for i in range(i_stop+1):
        overflow = checkMemory(mem_limit = 95)
        if overflow == True:
            break      

        

        #################################################
        #CALCULATION OF PHYSICAL QUANTITIES
        # except Husimi Phase, that's later during plotting
        #################################################
        #POSITION SPACE CALCULATIONS:
        #1. Calculate Total Density
        # rho = get_rho(chi,stars)
        rho_FDM = get_rho_FDM(chi)
        rho_part1,rho_part2 = get_rho_part1(stars),get_rho_part2(stars)
        rho = rho_FDM + rho_part1 + rho_part2

        #2. Calculate potential 
        phi = get_phi(rho)

        #3. Calculate Acceleration Field on Mesh:
        a_grid = -np.gradient(phi,dz)
        a_grid -= np.mean(a_grid)
        
        # ##########################################
        # DIAGNOSTICS + STORAGE
        K_star_fine_storage[i], W_star_fine_storage[i], W_2_star_storage[i] = get_stars_KW(stars, rho_part1, rho_part2, phi)
        
        if Num_stars!=0 and (i == 0 or i == i_stop):
            if variable_mass[0] == True:
                K_array=[stars[0].get_K(),stars[1].get_K()]
                W_array=[stars[0].get_W(z,phi),stars[1].get_W(z,phi-np.min(phi))]
            else:
                K_array=stars.get_K()
                W_array=stars.get_W(z,phi-np.min(phi))
            if i==0:
                K_star_storage[0] = K_array
                W_star_storage[0] = W_array  
            if i==i_stop:
                K_star_storage[-1] = K_array
                W_star_storage[-1] = W_array  
        
        if i in snapshot_indices:
            if Num_bosons!=0:
                np.savetxt(f"chi_storage_{i}.csv", chi, delimiter=",")
            if variable_mass[0] == True:    
                np.savetxt(f"QPs_m_storage_{i}.csv", stars[0].mass, delimiter=",")
                np.savetxt(f"QPs_x_storage_{i}.csv", stars[0].x, delimiter=",")
                np.savetxt(f"QPs_v_storage_{i}.csv", stars[0].v, delimiter=",")
                np.savetxt(f"stars_x_storage_{i}.csv", stars[1].x, delimiter=",")
                np.savetxt(f"stars_v_storage_{i}.csv", stars[1].v, delimiter=",")
            else: 
                np.savetxt(f"stars_x_storage_{i}.csv", stars.x, delimiter=",")
                np.savetxt(f"stars_v_storage_{i}.csv", stars.v, delimiter=",")    

        K_FDM_storage[i], W_FDM_storage[i] = get_FDM_KW(chi,rho_FDM,phi)
        
        # if i in track_snapshot_indices: # Happens only at snapshot indices    
        if i%(4*collapse_index) == 0: #Every 4 collapse times
            z_rms_storage, v_rms_storage = get_stars_rms(stars, z_rms_storage, v_rms_storage)
            
            R_half_mass = np.percentile(np.sort(np.abs(z_rms_storage)),50)
            R_half.append(R_half_mass)
        if history == True:
            acceleration_storage.append(a_grid)
        
        # st = process_time() 
        
        #################################################
        # PLOTTING
        # Plot everytime if sim_choice2 == 1
        # Plot only specific time steps if sim_choice2 == 2
        #################################################
        if absolute_PLOT == True: #go through options for plotting
            PLOT = False
            if sim_choice2 == 1:
                PLOT = True #always plot
            elif sim_choice2 == 2:
                if i in snapshot_indices: #check if time-step is correct one.
                    PLOT = True
    
                    
            if PLOT == True:
                part_centroid, fdm_centroid = main_plot(type,G_tilde,L,eta,
                z,dz,mu,chi,rho_FDM,rho_part1,rho_part2,
                stars,Num_bosons,Num_stars,dtau,i,
                x_min,x_max,v_min,v_max,
                y00_max,y10_max,y01_max,y11_max,
                a_max,
                Directory,folder_name, track_stars, track_centroid, variable_mass)
                
                if track_centroid == True:
                    if i == 0:
                        part_centroids = [part_centroid]
                        fdm_centroids = [fdm_centroid]
                    else:
                        part_centroids.append(part_centroid)
                        fdm_centroids.append(fdm_centroid)

        # et = process_time()
    
        ############################################################
        #EVOLVE SYSTEM (After calculations on the Mesh)
        ############################################################
        #1,2: Kick+Drift
        #FUZZY DM
        chi = FDM.kick(chi,phi/2,r,dtau/2)
        chi = FDM.drift(chi,r,dz,dtau)
        #PARTICLES
        g_interp = interp1d(z,a_grid)
        stars = kick(stars, g_interp, dtau)
        stars = drift(stars, dtau)

        #3: Re-update potential and acceleration fields
        rho = get_rho(chi,stars)
        phi = get_phi(rho)

        #4: KICK in updated potential
        #FUZZY DM
        chi = FDM.kick(chi,phi/2,r,dtau/2)
        #PARTICLES
        a_grid = -np.gradient(phi,dz)
        a_grid -= np.mean(a_grid)
        g_interp = interp1d(z,a_grid)
        stars = kick(stars, g_interp, dtau)
        
        time += dtau
        i += 1

        # print(et-st)

    # np.savetxt("chi_storage.csv", chi_storage, delimiter=",")
    # np.savetxt("stars_x_storage.csv", stars_x_storage, delimiter=",")
    # np.savetxt("stars_v_storage.csv", stars_v_storage, delimiter=",")
    np.savetxt("acceleration_storage.csv", acceleration_storage, delimiter=",")
    np.savetxt("R_half_storage.csv", R_half, delimiter=",")
    if track_centroid == False:
        part_centroids = []
        fdm_centroids = [] 
     
    return snapshot_indices, stars, chi, z_rms_storage, v_rms_storage, K_star_storage, W_star_storage, W_2_star_storage, K_star_fine_storage, W_star_fine_storage, part_centroids, fdm_centroids, K_FDM_storage, W_FDM_storage  


###########################################################
# FOR ANIMATION IN POSITION SPACE
###########################################################
def plot_save_waves(x,psi_s,phi_s,P_s,dt,Num_Particles,Directory,folder_name: str):
    os.chdir(Directory + "/" + folder_name)

    y_max = np.max([np.max(P) for P in P_s]) #y bounds. For density plots
        
    for i in range(len(P_s)):
        fig,ax = plt.subplots(1,2,figsize = (20,10))
        plt.suptitle(f"Time {round(dt*i,5)}".zfill(5))
        
        ax[0].plot(x,psi_s[i].real, label = "Re[$\\psi$]")
        ax[0].plot(x,psi_s[i].imag, label = "Im[$\\psi$]")
        ax[0].plot(x,phi_s[i],label = "Potential [Fourier perturbation]")
        ax[0].plot(x,np.absolute(psi_s[i])**2,label = "$|\\psi|^2$")
        ax[0].set_ylim([-np.max(np.absolute(psi_s[0])**2),np.max(np.absolute(psi_s[0])**2)])

        ax[0].set_xlabel("$z = x/L$")
        ax[0].legend()

        ax[1].plot(x,P_s[i],label = "Perturbation on $\\rho$")
        ax[1].set_ylim([-y_max, y_max] )

        ax[1].set_xlabel("$z = x/L$")
        ax[1].legend()
        
        #now save it as a .jpg file:
        folder = Directory + "/" + folder_name
        filename = 'ToyModelPlot' + str(i+1).zfill(4) + '.jpg';
        plt.savefig(folder + "/" + filename)  #save this figure (includes both subplots)
        
        plt.close() #close plot so it doesn't overlap with the next one


# Video Generating function
def generate_video(fourcc,Directory, folder_name: str, video_name: str,fps):
    os.chdir(Directory)
    image_folder = Directory +"/"+folder_name

    images = [img for img in sorted(os.listdir(image_folder))
            if img.endswith(".jpg") or
                img.endswith(".jpeg") or
                img.endswith("png")]
    
    # Array images should only consider
    # the image files ignoring others if any
    #print(images) 

    frame = cv2.imread(os.path.join(image_folder, images[0]))

    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape  

    fps = int(fps)#int(1/dt)
    #fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height)) 

    # Appending the images to the video one by one
    for image in images: 
        video.write(cv2.imread(os.path.join(image_folder, image))) 
    
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated
  
def animate(fourcc,Directory: str,folder_name: str, video_name: str, dt):
    path = Directory +"/"+folder_name
    
    # Folder which contains all the images
    # from which video is to be generated
    os.chdir(path)  
    
    mean_height = 0
    mean_width = 0
    
    num_of_images = len(os.listdir(path))
    #print(num_of_images)
    
    for file in os.listdir(path):
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            im = Image.open(os.path.join(path, file))
            width, height = im.size
            mean_width += width
            mean_height += height
            # im.show()   # uncomment this for displaying the image
    
    # Finding the mean height and width of all images.
    # This is required because the video frame needs
    # to be set with same width and height. Otherwise
    # images not equal to that width height will not get 
    # embedded into the video
    mean_width = int(mean_width / num_of_images)
    mean_height = int(mean_height / num_of_images)
    
    # print(mean_height)
    # print(mean_width)
    
    # Resizing of the images to give
    # them same width and height 
    for file in os.listdir(path):
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"):
            # opening image using PIL Image
            im = Image.open(os.path.join(path, file)) 
    
            # im.size includes the height and width of image
            width, height = im.size   
            #print(width, height)
    
            # resizing 
            im = im.convert('RGB')
            imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS) 
            imResize.save( file, 'JPEG', quality = 1080) # setting quality
            # printing each resized image name
            #print(im.filename.split('\\')[-1], " is resized") 
    
    # Calling the generate_video function
    generate_video(fourcc,Directory, folder_name, video_name,dt)


###############################################

