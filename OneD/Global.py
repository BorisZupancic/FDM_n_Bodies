from re import M
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
    #process = psutil.Process(os.getpid())
    memoryUsage = psutil.virtual_memory().percent
    #print(f"Memory Usage = {memoryUsage} %")
    overflow = False
    if memoryUsage > mem_limit:
        print(f"Memory usage exceeded budget of {mem_limit} percent.")
        overflow = True
    return overflow


def Startup(hbar,L_scale,v_scale):

    M_scale = L_scale*v_scale**2
    
    print("")
    print("Choose a (non-dimensional) box length:")
    L = float(input())
    print("")

    Total_mass = 1
    print("Choose percentage (as a decimal) of FDM (by mass)")
    percent_FDM = float(input())
    percent_Particles = 1 - percent_FDM
    print(f"Fraction of Particles (by mass) = {percent_Particles}")
    
    print("")
    if percent_FDM != 0:
        print("Choose a FDM fuzziness.")
        print("Input a FDM velocity dispersion, Characteristic size, and de Broglie wavelength ratio to Characteristic size:")
        v_FDM = float(input()) #/v_scale
        R = float(input()) #/L_scale
        lambda_deB = float(input()) #/R
        r = (1/(4*np.pi))*v_FDM*R*lambda_deB
        print(f"lambda_deB = {R*lambda_deB}")
        #r = float(input())
        print(f"Fuzziness: r = {r}")
        m = hbar/(2*r*v_scale*L_scale)
        #Calculate dimensional mass:
        mu = m/M_scale
        #print(f"This gives non-dim mass: mu = {mu}")
        print(f"Mass mu = {mu}, m = mu*M = {m}")
    elif percent_FDM == 0:
        r = 0.5
        mu = 1 #set as default
        R = None
        lambda_deB = None 


    print("")
    if percent_FDM != 1:
        print("How many particles?")
        Num_stars = int(input())

            
        sigma = Total_mass*(1-percent_FDM) / Num_stars
        
        FDM_mass = Total_mass*percent_FDM #int(input())
        Num_bosons = FDM_mass/mu
    else:
        Num_stars = 0
        sigma = 0

        FDM_mass = Total_mass
        Num_bosons = FDM_mass/mu
    
    print(f"Num_stars = {Num_stars}")
    print(f"sigma = {sigma}")
    
    print(f"Num_Bosons = {Num_bosons}")
    print(f"mu = {mu}")
    
    return L, mu, Num_bosons, r, lambda_deB, R, sigma, Num_stars

def gaussian(x,b,std):
    return np.exp(-(x-b)**2/(2*std**2))/(np.sqrt(2*np.pi)*std)

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
    #num = np.sum(rho)
    #print(f"num = {num}")

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
    
    #phi = phi/len(phi)
    #phi = phi*L

    # rho_new = np.gradient(np.gradient(phi,dz),dz)/(4*np.pi)
    # M1 = dz*np.sum(rho)
    # M2 = dz*np.sum(rho_new)

    # print(f"M1 = {M1}")
    # print(f"M2 = {M2}")
    # print(f"M2/M1 = {M2/M1}")
    
    # fig, ax = plt.subplots(1,5, figsize=(30,5))
    # ax[0].plot(rho, label = "$\\rho$")
    # ax[1].plot(p_n, label = "$\\hat{\\rho}$")
    # ax[2].plot(phi_n, label = "$\\hat{\\phi}$")
    # ax[3].plot(phi, label = "$\\phi$")
    # ax[4].plot(np.gradient(np.gradient(phi,dz),dz)/(4*np.pi), label = "$\\rho$ from Poisson")
    # for i in range(5):
    #     ax[i].legend()
    # plt.show()

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
                z,dz,chi,rho_FDM,rho_part,
                stars,Num_bosons,Num_stars,dtau,i,
                x_min,x_max,v_min,v_max,
                y00_max,y10_max,y01_max,y11_max,
                a_max,max_F,
                Directory,folder_name,track_stars, track_centroid = False, variable_mass=[False]):

    layout = [['upper left', 'upper right', 'far right'],
                        ['lower left', 'lower right', 'far right']]

    fig, ax = plt.subplot_mosaic(layout, constrained_layout = True)
    fig.set_size_inches(20,10)
    plt.suptitle("Time $\\tau = $" +f"{round(dtau*i,5)}".zfill(5), fontsize = 20)    
    
    ##############################################
    #ACCELERATIONS
    Part_force = -gradient(fourier_potential(rho_part,L,type = type, G_tilde = G_tilde),L,type = type)
    FDM_force = -gradient(fourier_potential(rho_FDM,L,type = type, G_tilde = G_tilde),L,type = type)
    ax['far right'].plot(z, Part_force, label = "Particle Contribution")
    ax['far right'].plot(z, FDM_force, label = "FDM Contribution")
    ax['far right'].set_ylim(-a_max,a_max)
    ax['far right'].set_title("Force contributions",fontsize = 15)
    ax['far right'].set_xlabel("$z = x/L_s$")
    ax['far right'].set_ylabel("Acceleration Field (code units)")
    ax['far right'].legend(fontsize = 20)
    
    # FDM
    phi_FDM = fourier_potential(rho_FDM,L,type = type, G_tilde = G_tilde)
    ax['upper left'].plot(z,phi_FDM,label = "$\\varphi_{FDM}$")
    ax['upper left'].plot(z,rho_FDM,label = "$\\rho_{FDM} = \\mu|\\chi|^2$")
    ax['upper left'].set_ylim([-y00_max, y00_max] )
    ax['upper left'].set_xlabel("$z = x/L_s$")
    ax['upper left'].legend(fontsize = 15)
    ax['upper left'].set_title("Non-Dimensional Densities and Potentials",fontsize = 15)
    
    if Num_bosons !=0:
        #PHASE SPACE CALCULATION:
        F = FDM.Husimi_phase(chi,z,dz,L,eta)
        # if i == 0:
        #     max_F = np.max(F)/2
        ax['upper right'].set_title("Phase Space Distributions", fontsize = 15)
        
        im = ax['upper right'].imshow(F,extent = (x_min,x_max,v_min,v_max),cmap = cm.coolwarm, norm = Normalize(0,max_F), aspect = (x_max-x_min)/(2*y01_max)) #LogNorm(0,max_F)
        ax['upper right'].set_xlim(x_min,x_max)
        ax['upper right'].set_ylim(-y01_max,y01_max) #[v_min,v_max])
        ax['upper right'].set_xlabel("$z = x/L_s$")
        ax['upper right'].set_ylabel("Velocity (code units)")
        #ax['upper right'].colorbar()
        #divider = make_axes_locatable(ax["upper right"])
        #cax = divider.new_horizontal(size = '5%',pad = 0.05, pack_start = True)
        #fig.add_axes(cax)
        #fig.colorbar(mappable = im, cax = cax, ax=ax["upper right"],shrink = 0.75)
    ##############################################3
    # PARTICLES
    phi_part = fourier_potential(rho_part,L,type = type, G_tilde = G_tilde)
    ax['lower left'].plot(z,phi_part,label = "$\\varphi_{Particles}$")
    ax['lower left'].plot(z,rho_part,label = "$\\rho_{Particles}$")
    ax['lower left'].set_ylim(-y10_max,y10_max)
    ax['lower left'].legend(fontsize = 15)

    #Plot the Phase Space distribution
    if variable_mass[0] == True:
        fraction = variable_mass[1]
        num_to_change = int(np.floor(fraction*len(stars)))
    
        x1_s = np.array([star.x for star in stars[num_to_change:]])
        v1_s = np.array([star.v for star in stars[num_to_change:]]) 
        ax['lower right'].scatter(x1_s,v1_s,s = 1,c = 'blue', label = "Lighter Particles")

        x2_s = np.array([star.x for star in stars[0:num_to_change]])
        v2_s = np.array([star.v for star in stars[0:num_to_change]]) 
        ax['lower right'].scatter(x2_s,v2_s,s = 8,c = 'red', label = "Heavier Particles")
        
        ax['lower right'].set_ylim(-y11_max,y11_max)
        ax['lower right'].set_xlim(-L/2,L/2)
        ax['lower right'].legend(fontsize = 15)
        ax['lower right'].set_ylabel("Velocity (code units)")
    else:
        x_s = np.array([star.x for star in stars])
        v_s = np.array([star.v for star in stars])
        xy = np.vstack([x_s,v_s])
        z = gaussian_kde(xy)(xy)
        ax['lower right'].scatter(x_s,v_s,s = 1,c=z,label = "Particles")
        ax['lower right'].set_ylim(-y11_max,y11_max)
        ax['lower right'].set_xlim(-L/2,L/2)
        ax['lower right'].legend(fontsize = 15)

    #ADDITIONAL:
    # Plotting the paths of those select stars
    if track_stars == True and Num_stars >= 5:
        #E_array = np.array([])
        for j in range(5):
            ax['lower right'].scatter(stars[j].x, stars[j].v, c = 'k', s = 50, marker = 'o')
        #    E_array = np.append(E_array,stars[j].v**2)
        #E_storage = np.append(E_storage,[E_array])
        
    #PLOT CENTROID IN PHASE SPACE
    if Num_stars != 0:#only calculate if there are stars
        centroid_z = 0
        centroid_v = 0
        for star in stars:
            centroid_z += star.x
            centroid_v += star.v
        part_centroid_z = centroid_z / Num_stars    
        part_centroid_v = centroid_v / Num_stars
        
        ax['lower right'].scatter(part_centroid_z,part_centroid_v,s = 100,c = "r",marker = "o")
        part_centroid = np.array([centroid_z,centroid_v])
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
        
    #now save it as a .jpg file:
    folder = Directory + "/" + folder_name
    filename = 'Plot' + str(i).zfill(4) + '.jpg';
    plt.savefig(folder + "/" + filename)  #save this figure (includes both subplots)
    plt.clf()
    plt.cla()
    plt.close(fig) #close plot so it doesn't overlap with the next one

    if track_centroid == True:
        return part_centroid, fdm_centroid

def gaussianICs(z, L, mu, Num_bosons, sigma, Num_stars, v_s, L_s):
    ########################################################
    # INITIAL SETUP
    ########################################################
    # FOR THE FDM
    #Set an initial wavefunction
    b=0
    if Num_bosons != 0:
        print("Choose the standard deviation of the initial FDM distribution (as a fraction of the box width):")
        std = float(input())
    else:
        std = 0.1 #value doesn't matter (just not 0)
    std=std*L
    psi = np.sqrt(gaussian(z,b,std)*Num_bosons)#*Num_particles / (L**3))
    chi = psi*L_s**(3/2)

    ####################################################
    # FOR THE PARTICLES
    #Set initial distribution on grid
    b = 0 #center at zero
    if Num_stars != 0:
        print("Choose the standard deviation of the Particle system (as fraction of the box width):")
        std = float(input())
    else:
        std = 0.1 #value doesn't matter (just not 0)
    std = std*L 
    z_0 = np.random.normal(b,std,Num_stars) #initial positions sampled from normal distribution
    stars = [NB.star(i,sigma,z_0[i],0) for i in range(len(z_0))] #create list of normally distributed stars, zero initial speed
    
    #reposition stars if they were generated outside the box
    for star in stars:
        if np.absolute(star.x) > L/2:
                star.reposition(L)

    # Reposition the center of mass
    grid_counts = NB.grid_count(stars,L,z)
    if Num_stars != 0: 
        centroid_z = 0
        for j in range(len(grid_counts)):
            centroid_z += z[j]*grid_counts[j]
        centroid_z = centroid_z / Num_stars

        for star in stars:
            star.x = star.x - centroid_z #shift
            star.reposition(L) #reposition

    #Calculate Collapse Time:
    rho_FDM = mu*np.absolute(chi)**2 #just norm-squared of wavefunction
    
    #Calculate distribution on Mesh
    if Num_stars !=0:
        rho_part = NB.particle_density(stars, L, z, variable_mass = False)
    else:
        rho_part = np.zeros_like(z)

    rho = rho_FDM+rho_part
    rho_avg = np.mean(rho)
    print(rho_avg)
    tau_collapse = 1/(rho_avg)**0.5
    print(f"(Non-dim) Collapse time: {tau_collapse}")
    t_dynamical = tau_collapse

    return stars, chi, t_dynamical

def sine2_ICs(z, L, Num_bosons, sigma, Num_stars, v_s, L_s):
    dist = np.cos(2*np.pi*z/L)**2
    dist[:len(dist)//4] = 0
    dist[3*len(dist)//4:] = 0
    
    plt.plot(dist)
    plt.show()
    # FOR THE FDM
    #Set an initial wavefunction
    chi = L_s*dist*np.sqrt(Num_bosons)
    
    ####################################################
    # FOR THE PARTICLES
    #Define CDF:
    dz = z[1]-z[0]
    cdf = np.zeros_like(z)
    for i in range(len(cdf)):
        val = np.sum(dist[:i])*dz
        cdf[i] = val
    plt.plot(z,cdf,label="CDF")

    #sample uniformly between 0 and 1
    y_0 = np.random.uniform(0,1,Num_stars)
    #invert:
    z_0 = []
    for i in range(len(z)-1):
        for y in y_0:
            if cdf[i] < y and y < cdf[i+1]:
                z_0.append(z[i])
                break
            

    stars = [NB.star(i,sigma,z_0[i],0) for i in range(len(z_0))] #create list of normally distributed stars, zero initial speed
    
    #reposition stars if they were generated outside the box
    for star in stars:
        if np.absolute(star.x) > L/2:
                star.reposition(L)

    # Reposition the center of mass
    grid_counts = NB.grid_count(stars,L,z)
    if Num_stars != 0: 
        centroid_z = 0
        for j in range(len(grid_counts)):
            centroid_z += z[j]*grid_counts[j]
        centroid_z = centroid_z / Num_stars

        for star in stars:
            star.x = star.x - centroid_z #shift
            star.reposition(L) #reposition

    return stars, chi

##########################################
#my own Spitzer ICs, based off Larry's:
def BZ_SpitzerICs(Num_stars, z, E0, sigma, f0, mu = None, Num_bosons = None, r = None):
    def density(psi):
        rho = np.zeros_like(psi)
        tm2 = (E0 - psi)/sigma**2
        coef = 2**1.5*f0*sigma
        rho[tm2>0] = coef*(np.sqrt(np.pi)/2.*np.exp(tm2[tm2>0])*erf(np.sqrt(tm2[tm2>0]))-np.sqrt(tm2[tm2>0]))
        return rho

    def derivs(y):
        dy = np.zeros(2)
        dy[0] = y[1] #first derivative
        dy[1] = 4.*np.pi*density(y[0]) #second derivative
        return dy

# STEP 1: SOLVE FOR rho, phi VIA RK4
    #initial conditions:
    y = np.zeros(2) #phi = 0, dphi/dt = 0 

    z_original = np.copy(z)
    dz = z[1]-z[0]
    N = len(z)
    z = np.zeros(len(z)//2)
    phi = np.zeros(len(z))
    rho = np.zeros(len(z))
    rho[0] = density(phi[0])
    i = 0
    while rho[i] > 0 and i<len(z)-1:
        k1 = derivs(y)
        k2 = derivs(y + dz*k1/2.)
        k3 = derivs(y + dz*k2/2.)
        k4 = derivs(y + dz*k3)
        y = y + (k1 + 2.*k2 + 2.*k3 + k4)*dz/6.
        
        i+=1
        z[i] = z[i-1]+dz
        phi[i] = y[0]
        rho[i] = density(y[0])

    imax = i    
    rho = rho[:imax]
    phi = phi[:imax]
    z = z[:imax]
    
    # check normalization:
    M = 2*np.sum(rho)*dz
    print(f"Mass from distribution function: M = {M}")
     
    L = 2*(z[-1]-z[0])
    z_new = np.append(-z[::-1],z)
    rho_new = np.append(rho[::-1],rho)
    phi_new = np.append(phi[::-1],phi)
    
    z_long = np.linspace(-L,L,2*len(z_new)-1)
    G = 0.5*np.abs(z_long)
    G_tilde = np.fft.fft(G)

    my_phi = fourier_potential(rho_new,L,type='Isolated', G_tilde = G_tilde)
    my_phi = my_phi - np.min(my_phi)

    rhomax = rho[0]
    zmax = z[imax-1]
    from scipy.interpolate import interp1d
    rhointerp = interp1d(z,rho)
    phiinterp = interp1d(z,phi)
    print ('cut-off in z', zmax)

#STEP 2: Randomly Sample Stars (if applicable)
    if Num_stars !=0:
        print("-----Sampling Stars-----")
        # generate initial conditions by sampling the density to get
        # distribution in z and then sampling the DF at fixed z to get w
        xIC = np.zeros(Num_stars)
        vIC = np.zeros(Num_stars)
        for i in range(Num_stars):
            rho = 0.
            rtmp = 1.
            while rho < rtmp:
                xtmp = np.random.uniform(-zmax,zmax)
                rho = rhointerp(np.abs(xtmp))
                rtmp = np.random.uniform(0,rhomax) 
            xIC[i] = xtmp
            p = phiinterp(np.abs(xIC[i]))
            vmax = np.sqrt(2.*(E0-p))
            f = 0
            ftmp = 1.
            while f < ftmp:
                fmax = np.exp((E0-p)/sigma**2) - 1.
                vtmp = np.random.uniform(-vmax,vmax)
                etmp = vtmp**2/2. + p
                f = np.exp((E0-etmp)/sigma**2) - 1.
                ftmp = np.random.uniform(0,fmax)
            vIC[i] = vtmp

        m = M/Num_stars
        stars = [NB.star(0,m,x,v) for x,v in zip(xIC,vIC)]
        
        #re-center position and velocity centroids:
        print("Re-centering position and velocity centroids.")
        z_centroid = np.mean([star.x for star in stars])
        v_centroid = np.mean([star.v for star in stars])
        print(f"z_centroid = {z_centroid}, v_centroid = {v_centroid}")
        for star in stars:
            star.x += -z_centroid
            star.v += -v_centroid
        z_centroid = np.mean([star.x for star in stars])
        v_centroid = np.mean([star.v for star in stars])
        print(f"z_centroid = {z_centroid}, v_centroid = {v_centroid}")
        
        v_rms = np.std([star.v for star in stars])
        z_rms = np.std([star.x for star in stars])
        print(f"z_rms = {z_rms}")
        print(f"v_rms = {v_rms}")
        t_dynamical = z_rms/v_rms
        print(f"t_dynamical = {t_dynamical}")
        
    else:
        stars = []

#STEP 3: Create FDM wavefunction (if applicable)
    if Num_bosons!=0:
        print("------Sampling FDM------")
        #Function that will return N by N array for DF:
        def DF(phi, E0,sigma,f0):
            N = len(phi)
            v_max = np.sqrt(2*E0)  
            v = np.linspace(-v_max, v_max, N)
            # Fill up f
            # z -> columns, v -> rows
            A = np.exp(E0/sigma**2)
            f = np.ndarray((N,N))
            for i in range(N):
                pot = phi[i]
                for j in range(N):
                    vv = v[j]
                    E = 0.5*vv**2 + pot
                    if E <= E0:
                        B = np.exp(-E/sigma**2)
                        f[j,i] = f0*(A*B-1)
                    else: 
                        f[j,i] = 0
            return f, v
        
        N_old = np.copy(N)
        z = z_new
        rho = rho_new
        phi = phi_new
        
        #2. Create phase space distribution:
        N = len(phi)
        v_max = np.sqrt(2*E0)  
        v = np.linspace(-v_max, v_max, N)
        
        f, v = DF(phi, E0,sigma,f0)
        dv = v[1]-v[0]
        
        v_dist = dz*np.sum(f, axis = 1)
        z_dist = dv*np.sum(f, axis = 0)
        v_rms = np.sqrt(dv*np.sum(v_dist*v**2))
        z_rms = np.sqrt(dz*np.sum(z_dist*z**2))
        print(f"z_rms = {z_rms}")
        print(f"v_rms = {v_rms}")
        t_dynamical = z_rms/v_rms
        print(f"t_dynamical = {t_dynamical}")
        
        #3. Make wavefunction
        vmax = np.sqrt(2.*E0)
        pmax = mu*vmax
        p = np.linspace(-pmax,pmax,N)
        chi = np.zeros(N, dtype = complex)
        thetas = np.random.uniform(0,2*np.pi,N)
        R_s = np.exp(1j*thetas)
        for i in range(N):
            chi[i] = np.sum(R_s*np.sqrt(f[:,i])*np.exp(1j*p*z[i]))
        Normalization = np.sqrt( dz*np.sum(rho) / (dz*np.sum(np.abs(chi)**2)) / mu) 
        chi = chi * Normalization

        print(f"Area under mu|chi|^2: {mu*dz*np.sum(np.abs(chi)**2)}")
        print(f"Area under rho (from DF): {dz*np.sum(rho)}")

        #3b) corrections to momentum/velocity profile:
        print("Correcting center of momentum offset ...")
        k_mean = np.sum(np.conj(chi)*(-1j)*np.gradient(chi,dz)) / np.sum(np.abs(chi)**2)
        print(f"k_mean={k_mean}")
        print(f"v_mean = {k_mean/mu}")
        print("-->")
        
        T_drift = dz*mu/np.abs(k_mean)
        while t_dynamical/T_drift >= 10**(-3):
            chi = chi*np.exp(-1j*k_mean*z)
            k_mean = np.sum(np.conj(chi)*(-1j)*np.gradient(chi,dz)) / np.sum(np.abs(chi)**2)
            T_drift = mu*dz/np.abs(k_mean)
        
        print(f"k_mean={k_mean}")
        print(f"v_mean = {k_mean/mu}")
            
        #Extend wavefunction to entire box:
        N_add = int(N_old-len(chi))
        chi = np.append(np.zeros(N_add//2),chi)
        chi = np.append(chi,np.zeros(N_add//2))
        rho = mu*np.abs(chi)**2
    else:
        chi = np.zeros(N)

    if Num_bosons!=0 and Num_stars!=0:
        chi = chi / np.sqrt(2)
        for star in stars:
            star.mass = star.mass / 2

    return stars, chi, t_dynamical

def run_FDM_n_Bodies(sim_choice2, dynamical_times, t_dynamical, bc_choice, z, L, dz, 
                    mu, Num_bosons, r, chi, 
                    sigma, stars, 
                    v_s, L_s, 
                    Directory, folder_name, 
                    absolute_PLOT = True, track_stars = False, track_stars_rms = False, track_centroid = False, fixed_phi = False,
                    track_FDM = False, variable_mass = False):
    
    #########################################
    #RETRIEVE INFO FROM INITIAL STARTUP/CONDITIONS
    #Re-calcualte Mass and Time scales
    M_s = v_s**2 * L_s
    T_s = L_s / v_s

    Num_stars = len(stars)

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

    #check whether to run FDM or NBody or both
    # if Num_bosons == 0:
    #     sim_choice1 = 2 #revert to N-Body only calculation
    # elif Num_stars == 0: 
    #     sim_choice1 = 1 #revert to FDM only calculation
    # else:
    #     sim_choice1 = 3 #FDM and N-Body simultaneously

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
        phi = 0.5*sigma*((z**2) / (L/2)**2 - 1)  #equal to 0 at boundaries 

    ###################################################
    #PHASE SPACE STUFF
    N = len(z)
    eta=(z[-1]-z[0])/np.sqrt(np.pi*len(chi)/2)
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
    y00_max = np.max(rho_FDM)*10
    y10_max = np.max(rho_part)*10

    if Num_stars == 0:
        y10_max = y00_max
    elif Num_bosons == 0:
        y00_max = y10_max
    
    Rho_avg = M_s*np.mean(rho)/L_s
    T_collapse = 1/(Rho_avg)**0.5
    if Num_stars !=0:
        y01_max = 2*np.max([star.v for star in stars])
    else:
        y01_max = v_max/3 
    y11_max = y01_max
    
    ####################################################
    #PRE-LOOP TIME-SCALE SETUP
    collapse_index= 10
    dtau = (collapse_index**(-1))*t_dynamical 
    tau_stop = dynamical_times*t_dynamical 
    i_stop = dynamical_times * 10
    indices = [0]
    x = 0 
    while 2**x <= dynamical_times:
        indices.append(2**x)
        x+=1
    if sim_choice2 == 1:
        tau_stop = t_dynamical*2 #over-ride previous tau_stop
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
            rho_part = NB.particle_density(stars, L, z, variable_mass)
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
        # Tracking Some stars
        # This is independant of plotting
        if track_stars == True and Num_stars >= 5:
            K_5stars = np.array([])
            W_5stars = np.array([])
            for j in range(5):
                star = stars[j]
                W = star.get_W(z,phi,L) #Find the potential energy:
                
                K = 0.5*star.mass*star.v**2 #Find kinetic energy

                K_5stars = np.append(K_5stars,K)
                W_5stars = np.append(W_5stars,W)
            if i == 0:
                K_5stars_storage = np.array([K_5stars])
                W_5stars_storage = np.array([W_5stars])
            else:
                K_5stars_storage = np.append(K_5stars_storage,[K_5stars],axis = 0)
                W_5stars_storage = np.append(W_5stars_storage,[W_5stars],axis = 0)
        else:
            K_5stars_storage = [] #None
            W_5stars_storage = [] #None

        #Tracking energies of ALL stars
        # Happens only at snapshot indices
        if track_stars == True:
            if i in track_snapshot_indices:
                K_array = np.array([])
                W_array = np.array([])
                
                for j in range(len(stars)):
                    star = stars[j]
                    W = star.get_W(z,phi,L)
                    K = 0.5*star.mass*star.v**2

                    K_array = np.append(K_array,K)
                    W_array = np.append(W_array,W)
                if i == 0:
                    K_star_storage = np.array([K_array])
                    W_star_storage = np.array([W_array])
                else:
                    K_star_storage = np.append(K_star_storage,[K_array],axis = 0)
                    W_star_storage = np.append(W_star_storage,[W_array],axis = 0)
            
            K_star_fine_storage = [] #None
            W_star_fine_storage = [] #None
            # if True: #i<=2500:
            #     K_array = np.array([])
            #     W_array = np.array([])
                
            #     for j in range(len(stars)):
            #         star = stars[j]
            #         W = star.get_W(z,phi,L)
            #         K = 0.5*star.mass*star.v**2

            #         K_array = np.append(K_array,K)
            #         W_array = np.append(W_array,W)
            #     if i == 0:
            #         K_star_fine_storage = np.array([K_array])
            #         W_star_fine_storage = np.array([W_array])
            #     else:
            #         K_star_fine_storage = np.append(K_star_fine_storage,[K_array],axis = 0)
            #         W_star_fine_storage = np.append(W_star_fine_storage,[W_array],axis = 0)

        else:
            K_star_storage = [] #None
            W_star_storage = [] #None

            K_star_fine_storage = [] #None
            W_star_fine_storage = [] #None
            #E_storage = None #to go in the main_plot loop

        if track_stars_rms == True:
            if i in track_snapshot_indices:
                z_rms = np.sqrt(np.mean([star.x**2 for star in stars]))
                v_rms = np.sqrt(np.mean([star.v**2 for star in stars]))
    
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
            if i in track_snapshot_indices:
                    W = 0.5*np.sum(rho_FDM*phi)*dz  
                    #Kinetic Energy:
                    chi_tilde = np.fft.fft(chi)
                    k = 2*np.pi*np.fft.fftfreq(len(chi),dz)
                    K = (2/(N**2))*r*np.sum(k**2 * np.absolute(chi_tilde)**2)
        
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
                if i == 0:
                    F = FDM.Husimi_phase(chi,z,dz,L,eta)
                    max_F = np.max(F)/2
                
                part_centroid, fdm_centroid = main_plot(type,G_tilde,L,eta,
                z,dz,chi,rho_FDM,rho_part,
                stars,Num_bosons,Num_stars,
                dtau,i,
                x_min,x_max,v_min,v_max,
                y00_max,y10_max,y01_max,y11_max,
                a_max,max_F,
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
        g_s = g_interp([star.x for star in stars])

        for star, g in zip(stars,g_s):
            star.kick_star(g,dtau/2)
            star.drift_star(dtau)

            #corrective maneuvers on star position
            if bc_choice == 2:
                if np.absolute(star.x) > L/2:
                    star.reposition(L)
            elif bc_choice == 1:
                pass

        #3 Re-update potential and acceleration fields
        #Calculate Particle distribution on Mesh
        if Num_stars !=0:
            rho_part = NB.particle_density(stars, L, z, variable_mass)
        else:
            rho_part = np.zeros_like(z)
        
        #Add the density from the FDM
        rho_FDM = mu*np.absolute(chi)**2 
        rho = rho_FDM + rho_part
        #Calculate potential 
        if fixed_phi == False: #if True, potential is fixed
            phi = fourier_potential(rho,L, type = type, G_tilde = G_tilde)
        #Calculate Acceleration Field on Mesh:
        a_grid = NB.acceleration(phi,L,type = type) 

        #4. KICK in updated potential
        #FUZZY DM
        chi = FDM.kick(chi,phi/2,r,dtau/2)

        #PARTICLES
        a_grid = NB.acceleration(phi,L,type = type) 
        g_interp = interp1d(z,a_grid)
        g_s = g_interp([star.x for star in stars])

        #g = NB.accel_funct(a_grid,L,dz,type=type)
        for star, g in zip(stars,g_s):
            star.kick_star(g,dtau/2)
            
        time += dtau
        i += 1
    
    if track_centroid == False:
        part_centroids = []
        fdm_centroids = [] 
     
    return snapshot_indices, stars, chi, z_rms_storage, v_rms_storage, K_star_storage, W_star_storage, K_star_fine_storage, W_star_fine_storage, K_5stars_storage, W_5stars_storage, part_centroids, fdm_centroids, K_FDM_storage, W_FDM_storage  


def kick_n_drift_1(star, g, L,bc_choice,dtau):
    #print(g)
    star.kick_star(g,dtau/2)
    star.drift_star(dtau)

    if bc_choice == 2:
        #corrective maneuvers on star position
        #(for positions that drift outside of the box...
        # must apply periodicity)
        if np.absolute(star.x) > L/2:
            star.reposition(L)
    elif bc_choice == 1:
        pass

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

