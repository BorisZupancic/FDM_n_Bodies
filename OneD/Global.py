from matplotlib.image import AxesImage
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf as erf
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import os
import cv2 
from PIL import Image
import OneD.FDM as FDM
import OneD.NBody as NB
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import multiprocessing as mp
import threading 

import psutil

def checkMemory(mem_limit):
    memoryUsage = psutil.virtual_memory().percent
    overflow = False
    if memoryUsage > mem_limit:
        print(f"Memory usage exceeded budget of {mem_limit} percent.")
        overflow = True
    return overflow

# def Startup(hbar,L_scale,v_scale):

#     M_scale = L_scale*v_scale**2
    
#     print("")
#     print("Choose a (non-dimensional) box length:")
#     L = float(input())
#     print(f"L={L}")
#     print("")

#     Total_mass = 1
#     print("Choose percentage (as a decimal) of FDM (by mass)")
#     percent_FDM = float(input())
#     percent_Particles = 1 - percent_FDM
#     print(f"Fraction of Particles (by mass) = {percent_Particles}")
    
#     print("")
#     if percent_FDM != 0:
#         print("Choose a FDM fuzziness.")
#         print("Input a desired FDM velocity dispersion, and de Broglie wavelength ratio to Characteristic size:")
#         # R = float(input()) #/L_scale
#         v_FDM = float(input()) #/v_scale
#         lambda_deB = float(input()) #/R
        
#         r = lambda_deB*v_FDM / (4*np.pi)
#         print(f"lambda_deB = {R*lambda_deB}")
#         print(f"Fuzziness: r = {r}")
#         m = hbar/(2*r*v_scale*L_scale)
#         #Calculate dimensional mass:
#         mu = m/M_scale
#         #print(f"This gives non-dim mass: mu = {mu}")
#         print(f"Mass mu = {mu}, m = mu*M = {m}")
#     elif percent_FDM == 0:
#         r = 0.5
#         mu = 1 #set as default
#         R = None
#         lambda_deB = None 


#     print("")
#     if percent_FDM != 1:
#         print("How many particles?")
#         Num_stars = int(input())

            
#         sigma = Total_mass*(1-percent_FDM) / Num_stars
        
#         FDM_mass = Total_mass*percent_FDM #int(input())
#         Num_bosons = FDM_mass/mu
#     else:
#         Num_stars = 0
#         sigma = 0

#         FDM_mass = Total_mass
#         Num_bosons = FDM_mass/mu
    
#     print(f"Num_stars = {Num_stars}")
#     print(f"sigma = {sigma}")
    
#     print(f"Num_Bosons = {Num_bosons}")
#     print(f"mu = {mu}")
    
#     return L, mu, Num_bosons, r, lambda_deB, v_FDM, sigma, Num_stars

# def gaussian(x,b,std):
#     return np.exp(-(x-b)**2/(2*std**2))/(np.sqrt(2*np.pi)*std)

#########################################################3
#FOURIER STUFF

def fourier_gradient(phi,length):
    n = len(phi)
    L = length 

    #1. FFT the density (perturbation)
    phi_n = np.fft.fft(phi,n) #fft for real input
    
    #2. Compute the fourier coefficients of phi
    grad_n = np.array([]) #empty for storage. Will hold the fourier coefficients
    for nn in range(len(phi_n)):
        k = 2*np.pi*nn/L
        val = -1j*phi_n[nn]*k
        grad_n = np.append(grad_n,val)
    
    #3. IFFT back to get Potential
    grad = np.fft.ifft(grad_n,n) #use Phi_n as Fourier Coefficients
    
    grad = np.real(grad)
    return grad

def gradient(f,length,type = 'Periodic'):
    if type == 'Isolated':
        N = len(f)
        dx = length/(N-1)
        grad = np.gradient(f,dx,edge_order = 2)
        return grad
    elif type == 'Periodic':
        #grad = fourier_gradient(f,length)
        #Have decided to do the same thing:
        N = len(f)
        dx = length/(N-1)
        grad = np.gradient(f,dx,edge_order = 2)
        
        return grad

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
    A FFT Poisson solver, taking as input a (non-dimensional) density array
    and the length of the interval on which it is defined.

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
    
#####################################################3
#Full Calculation/Simulation Functions

def main_plot(type,G_tilde,L,eta,
                z,dz,mu,chi,rho_FDM,rho_part,
                stars,Num_bosons,Num_stars,dtau,i,
                x_min,x_max,v_min,v_max,
                y00_max,y10_max,y01_max,y11_max,
                a_max,
                Directory = None,folder_name = None,track_stars = None, track_centroid = False, variable_mass=[False]):

    
    rho_part1, rho_part2 = NB.particle_density(stars,L,z,variable_mass)
    phi_part1 = fourier_potential(rho_part1,L,type = type, G_tilde = G_tilde)
    phi_part2 = fourier_potential(rho_part2,L,type = type, G_tilde = G_tilde)
     
    
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
    
    if variable_mass[0] == False:
        Part_force = -gradient(fourier_potential(rho_part2,L,type = type, G_tilde = G_tilde),L,type = type)
        ax['far right'].plot(z, Part_force, label = "Particle Contribution")
    else:
        Part1_force = -gradient(fourier_potential(rho_part1,L,type = type, G_tilde = G_tilde),L,type = type)
        Part2_force = -gradient(fourier_potential(rho_part2,L,type = type, G_tilde = G_tilde),L,type = type)
        ax['far right'].plot(z, Part1_force, label = "(Quasi) Particle Contribution")
        ax['far right'].plot(z, Part2_force, label = "(Light) Particle Contribution")
        ax['far right'].set_ylim(-a_max/2,a_max/2)
        
        ax['upper left'].plot(z,phi_part1, label = "$\\phi_{Quasi}$")
        ax['upper left'].plot(z, rho_part1, label = "$\\rho_{Quasi}$")
        
        ax['lower left'].plot(z,phi_part2,label = "$\\Phi_{Particles}$")
        ax['lower left'].plot(z,rho_part2,label = "$\\rho_{Particles}$")

        ax['upper right'].set_ylim(-y11_max,y11_max)
    
    # FDM
    if Num_bosons != 0:
        FDM_force = -gradient(fourier_potential(rho_FDM,L,type = type, G_tilde = G_tilde),L,type = type)
        ax['far right'].plot(z, FDM_force, label = "FDM Contribution")
        ax['far right'].set_ylim(-a_max,a_max)
    
        phi_FDM = fourier_potential(rho_FDM,L,type = type, G_tilde = G_tilde)
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
        ax['lower right'].scatter(x1_s,v1_s,s = 1,c = 'blue', label = "(Light) Particles")
        #Plot heavy particles 
        x2_s = stars[0].x
        v2_s = stars[0].v 
        ax['upper right'].scatter(x2_s,v2_s,s = 8,c = stars[0].mass, label = "(Quasi) Particles")
    else:
        x_s = stars.x
        v_s = stars.v
        ax['lower right'].scatter(x_s,v_s,c = stars.mass, s = 1,label = "Particles")
        
        
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
            part_centroid_z = np.mean(stars.x)
            part_centroid_v = np.mean(stars.v)
        
            #ax['lower right'].scatter(part_centroid_z,part_centroid_v,s = 100,c = "r",marker = "o")
            ax['lower right'].plot([-L/2,L/2],[part_centroid_v,part_centroid_v],"k--")
            ax['lower right'].plot([part_centroid_z,part_centroid_z],[-y11_max,y11_max],"k--")
        part_centroid = np.array([part_centroid_z,part_centroid_v])
    else:
        part_centroid = None

    if Num_bosons !=0:
        fdm_centroid_z = np.sum(np.conj(chi)*z*chi) / np.sum(np.abs(chi)**2)
        k_mean = np.sum(np.conj(chi)*(-1j)*np.gradient(chi,dz))
        mu = np.mean(rho_FDM/(np.absolute(chi)**2))
        fdm_centroid_v = k_mean/mu

        fdm_centroid = np.array([np.real(fdm_centroid_z),np.real(fdm_centroid_v)])
    else:
        fdm_centroid = None
    
    ax['upper left'].legend(fontsize = 15)
    ax['upper right'].legend(fontsize = 15)
    ax['lower left'].legend(fontsize = 15)
    ax['lower right'].legend(fontsize = 15) 
    ax['far right'].legend(fontsize = 20)

    ax['lower left'].set_ylim(-y10_max,y10_max)
    ax['upper left'].set_ylim([-y00_max, y00_max] )
    ax['lower right'].set_ylim(-y11_max,y11_max)
    ax['lower right'].set_xlim(-L/2,L/2)
    
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
                    track_FDM = False, variable_mass = False):
    
    #########################################
    #RETRIEVE INFO FROM INITIAL STARTUP/CONDITIONS
    #Re-calcualte Mass and Time scales
    M_s = v_s**2 * L_s
    T_s = L_s / v_s

    L = z[-1]-z[0]
    dz = z[1]-z[0]

    if variable_mass[0] == True:
        Num_stars = len(stars[0].x) + len(stars[1].x)
    else:
        Num_stars = len(stars.x)


    #Chooce Poisson-Solving Routine:
    if bc_choice == 1:
        type = 'Isolated'
    
        N = len(z)
        z_long = np.linspace(-L,L,2*N-1)
        G = 0.5*np.abs(z_long)
        G_tilde = np.fft.fft(G)
        
    elif bc_choice == 2:
        type = 'Periodic'
        G_tilde = None

    ##############################################
    # INITIAL SETUP
    #Calculate initial Density perturbation (non-dimensionalized and reduced)
    rho_FDM = mu*np.absolute(chi)**2 #just norm-squared of wavefunction
    psi = chi* L_s**(-3/2)

    #Check how it's normalized:
    print(f"integral of |chi|^2 : {np.sum(dz*rho_FDM)}")
    print(f"Numerically calculated integral of |psi|^2 : {np.sum(dz*np.absolute(psi)**2)}")

    m = mu*M_s
    
    #Calculate distribution on Mesh
    if Num_stars !=0:
        rho_part = NB.particle_density(stars, L, z, variable_mass)
    else:
        rho_part = np.zeros_like(z)
    
    #Calculate total density
    rho = rho_FDM + rho_part
    #check normalization
    print(f"Density Normalization Check: {np.sum(rho)*dz}")
    #Now option to freeze the potential:
    if fixed_phi == True:
        if variable_mass[0]==True:
            sigma = stars[0].mass
        else:
            sigma = stars.mass
        phi = 0.5*sigma*((z**2) / (L/2)**2 - 1)  #equal to 0 at boundaries 

    ###################################################
    #PHASE SPACE STUFF
    N = len(z)
    eta=(z[-1]-z[0])/np.sqrt(np.pi*len(chi)/2)
    #eta = (z[-1]-z[0]) / np.sqrt(2*np.pi*N)
    k = 2*np.pi*np.fft.fftfreq(len(z),dz)
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

    ##########################################################
    #PLOT AXIS LIMITS:
    #y0_max = np.max(phi)*1.5
    y00_max = np.max(rho_FDM)*3
    y10_max = np.max(rho_part)*3

    if Num_stars == 0:
        y10_max = y00_max
    elif Num_bosons == 0:
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
    indices = [0]
    x = 0 
    while 2**x <= dynamical_times:
        indices.append(2**x)
        x+=1
    if sim_choice2 == 1:
        dtau = 0.1*t_dynamical/5
        collapse_index = int(np.ceil(t_dynamical/dtau))
        i_stop = dynamical_times * collapse_index
        
        snapshot_indices = None 
            
    elif sim_choice2 == 2:
        snapshot_indices = np.multiply(collapse_index,indices)

        if dynamical_times*collapse_index not in snapshot_indices:
            snapshot_indices = np.append(snapshot_indices, dynamical_times*collapse_index)

    
        print(f"Snapshots at i = {snapshot_indices}")
    
    if track_stars == True or track_FDM == True:
        if sim_choice2 == 1:
            # collapse_index = int(np.floor(t_dynamical/dtau))
            snapshot_indices = np.multiply(collapse_index,indices)        
            track_snapshot_indices = snapshot_indices
        elif sim_choice2 == 2:
            track_snapshot_indices = snapshot_indices

    print(f"Sim will stop at tau = {tau_stop}")
    time = 0
    i = 0 #counter, for saving images
    
    if absolute_PLOT == True:
        os.chdir(Directory + "/" + folder_name) #Change Directory to where Image Folders are
    
    #while time <= tau_stop:
    while i <= i_stop:
        overflow = checkMemory(mem_limit = 95)
        if overflow == True:
            break      

        #################################################
        #CALCULATION OF PHYSICAL QUANTITIES
        # except Husimi Phase, that's later during plotting
        #################################################
        #POSITION SPACE CALCULATIONS:
        #1. Calculate Total Density
        #Calculate distribution on Mesh
        if Num_stars !=0:
            rho_part1, rho_part2 = NB.particle_density(stars, L, z, variable_mass)
            rho_part = rho_part1 + rho_part2
        else:
            rho_part = np.zeros_like(z)
        
        #Then add the density from the FDM
        rho_FDM = mu*np.absolute(chi)**2
        rho = rho_FDM + rho_part 

        #2. Calculate potential 
        if fixed_phi == False:
            phi = fourier_potential(rho,L, type = type, G_tilde = G_tilde)
        
        #3. Calculate Acceleration Field on Mesh:
        a_grid = NB.acceleration(phi,L,type = type) 
        
        ##########################################
        #Tracking energies of ALL stars
        if track_stars == True:
            if i in track_snapshot_indices: # Happens only at snapshot indices
                if variable_mass[0]==True:
                    K_array=[*stars[0].get_K(),*stars[1].get_K()]
                    W_array=[*stars[0].get_W(z,phi),*stars[1].get_W(z,phi)]

                else:
                    K_array = stars.get_K()
                    W_array = stars.get_W(z,phi)
            
                if i == 0:
                    K_star_storage = np.array([K_array])
                    W_star_storage = np.array([W_array])

                else:
                    K_star_storage = np.append(K_star_storage,[K_array],axis = 0)
                    W_star_storage = np.append(W_star_storage,[W_array],axis = 0)

            # K_star_fine_storage = [] #None
            # W_star_fine_storage = [] #None
            
            if variable_mass[0]==True:
                K_array=[stars[0].get_K(),stars[1].get_K()]
                W_array=[stars[0].get_W(z,phi),stars[1].get_W(z,phi)]

                K = np.array([np.sum(K_array[0]),np.sum(K_array[1])])
                W = 0.5*np.array([np.sum(W_array[0]),np.sum(W_array[1])])

                W_2 = np.array([dz * np.sum(rho_part1*z*np.gradient(phi,dz)),dz * np.sum(rho_part2*z*np.gradient(phi,dz))])
            else:
                K_array = stars.get_K()
                W_array = stars.get_W(z,phi)

                K = np.sum(K_array)
                W = 0.5*np.sum(W_array) 

                W_2 = dz * np.sum(rho_part*z*np.gradient(phi,dz))
            if i == 0:
                K_star_fine_storage = np.array([K])
                W_star_fine_storage = np.array([W])

                W_2_star_storage = np.array([W_2])
            else:
                K_star_fine_storage = np.append(K_star_fine_storage,[K], axis=0)
                W_star_fine_storage = np.append(W_star_fine_storage,[W],axis=0)

                W_2_star_storage = np.append(W_2_star_storage,[W_2],axis = 0)
            # K_star_storage = [] #None
            # W_star_storage = [] #None
            # W_2_star_storage = []
            
        else:
            K_star_storage = [] #None
            W_star_storage = [] #None

            K_star_fine_storage = [] #None
            W_star_fine_storage = [] #None

            W_2_star_storage = [] #None


        if track_stars_rms == True:
            if i in track_snapshot_indices:
                if variable_mass[0]==True:
                    z_rms = np.sqrt(np.mean([*stars[0].x**2,*stars[1].x**2]))
                    v_rms = np.sqrt(np.mean([*stars[0].v**2,*stars[1].v**2]))
                else:
                    z_rms = np.sqrt(np.mean(stars.x**2))
                    v_rms = np.sqrt(np.mean(stars.x**2))
    
                if i == 0:
                    z_rms_storage = np.array(z_rms)
                    v_rms_storage = np.array(v_rms)
                else:
                    z_rms_storage = np.append(z_rms_storage,z_rms)
                    v_rms_storage = np.append(v_rms_storage,v_rms)
        else:
            z_rms_storage = [] #None
            v_rms_storage = [] #None

        #########################################
        #FDM DIAGNOSTICS
        #Record Energies:
        if track_FDM == True:
            # if i in track_snapshot_indices:
            W = dz*np.sum(rho_FDM*z*np.gradient(phi,dz))
            
            chi_tilde = np.fft.fft(chi, norm = "ortho")
            k = 2*np.pi*np.fft.fftfreq(len(chi),dz)
            dk = k[1]-k[0]
            K = (L/N) * r*np.sum(k**2 * np.abs(chi_tilde)**2)
            
            if i == 0:
                W_FDM_storage = np.array([[W]])
                K_FDM_storage = np.array([[K]])
            else:
                W_FDM_storage = np.append(W_FDM_storage,[[W]],axis=0) 
                K_FDM_storage = np.append(K_FDM_storage,[[K]],axis=0)
        else:
            W_FDM_storage = [] #None
            K_FDM_storage = [] #None
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
                #check if time-step is correct one.
                if i in snapshot_indices:
                    PLOT = True
                #else:
                #   PLOT = False
                    
            if PLOT == True:
                if i == 0: #want to set a limit on the acceleration graph
                    z_long = np.linspace(-L,L,2*N-1)
                    G = 0.5*np.abs(z_long)
                    G_tilde = np.fft.fft(G)

                    Part_force = -gradient(fourier_potential(rho_part,L, type = type, G_tilde = G_tilde),L,type = type)
                    FDM_force = -gradient(fourier_potential(rho_FDM,L, type = type, G_tilde = G_tilde),L,type = type)

                    a1 = np.abs([np.max(Part_force),np.min(Part_force)])
                    a2 = np.abs([np.max(FDM_force),np.min(FDM_force)])
                    fig, ax = plt.subplots(1,2,figsize = (15,5))
                    ax[0].set_title("Fuzzy Dark Matter")
                    ax[0].plot(z,-gradient(fourier_potential(rho_FDM,L,type = 'Periodic'),L,type = 'Periodic'),label = 'Periodic')
                    ax[0].plot(z,fourier_potential(rho_FDM,L,type = 'Periodic'),label = 'Periodic')
                    ax[0].plot(z,fourier_potential(rho_FDM,L, type = 'Isolated', G_tilde = G_tilde),label = 'Isolated')
                    ax[0].plot(z,-gradient(fourier_potential(rho_FDM, L, type = 'Isolated', G_tilde = G_tilde),L,type = 'Isolated'),label = 'Isolated')
                    ax[0].set_xlabel("$z$")
                    ax[0].legend()
                    
                    ax[1].set_title("Particles")
                    ax[1].plot(z,-gradient(fourier_potential(rho_part,L,type = 'Periodic'),L,type = 'Periodic'),label = 'Periodic')
                    ax[1].plot(z,fourier_potential(rho_part,L,type = 'Periodic'),label = 'Periodic')
                    ax[1].plot(z,fourier_potential(rho_part,L, type = 'Isolated', G_tilde = G_tilde),label = 'Isolated')
                    ax[1].plot(z,-gradient(fourier_potential(rho_part,L, type = 'Isolated', G_tilde = G_tilde),L,type = 'Isolated'),label = 'Isolated')
                    ax[1].set_xlabel("$z$")
                    ax[1].legend()
                    plt.savefig("Initial Periodic vs Isolated")
                    plt.clf()
                    a_max = np.max(np.append(a1,a2))*2
                    print(f"a_max = {a_max}")
                     
                part_centroid, fdm_centroid = main_plot(type,G_tilde,L,eta,
                z,dz,mu, chi,rho_FDM,rho_part,
                stars,Num_bosons,Num_stars,
                dtau,i,
                x_min,x_max,v_min,v_max,
                y00_max,y10_max,y01_max,y11_max,
                a_max,
                Directory,folder_name,track_stars,track_centroid, variable_mass)
                
                if track_centroid == True:
                    if i == 0:
                        part_centroids = [part_centroid]
                        fdm_centroids = [fdm_centroid]
                    else:
                        part_centroids.append(part_centroid)
                        fdm_centroids.append(fdm_centroid)

        ############################################################
        #EVOLVE SYSTEM (After calculations on the Mesh)
        ############################################################
        #1,2: Kick+Drift

        #FUZZY DM
        chi = FDM.kick(chi,phi/2,r,dtau/2)
        chi = FDM.drift(chi,r,dz,dtau)

        #PARTICLES
        g_interp = interp1d(z,a_grid)
        if variable_mass[0] == True:
            g_0 = g_interp(stars[0].x)
            stars[0].kick(g_0,dtau/2)
            g_1 = g_interp(stars[1].x)
            stars[1].kick(g_1,dtau/2)

            stars[0].drift(dtau)
            stars[1].drift(dtau)
        else:
            g_s = g_interp(stars.x)
            stars.kick(g_s,dtau/2)
            stars.drift(dtau)

        #     #corrective maneuvers on star position
        #     if bc_choice == 2:
        #         if np.absolute(star.x) > L/2:
        #             star.reposition(L)
        #     elif bc_choice == 1:
        #         pass

        #3 Re-update potential and acceleration fields
        if fixed_phi == False: #if True, potential is fixed
            if Num_stars !=0:
                rho_part1, rho_part2 = NB.particle_density(stars, L, z, variable_mass)
                rho_part = rho_part1 + rho_part2
            else:
                rho_part = np.zeros_like(z)
            rho_FDM = mu*np.absolute(chi)**2 
            rho = rho_FDM + rho_part
            phi = fourier_potential(rho,L, type = type, G_tilde = G_tilde)
        
        #4. KICK in updated potential
        #FUZZY DM
        chi = FDM.kick(chi,phi/2,r,dtau/2)

        #PARTICLES
        a_grid = NB.acceleration(phi,L,type = type) 
        g_interp = interp1d(z,a_grid)
        if variable_mass[0] == True:
            g_0 = g_interp(stars[0].x)
            stars[0].kick(g_0,dtau/2)
            
            g_1 = g_interp(stars[1].x)
            stars[1].kick(g_1,dtau/2)
        else:
            g_s = g_interp(stars.x)
            stars.kick(g_s,dtau/2)

        time += dtau
        i += 1
    
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

