#from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 
from PIL import Image
import OneD.WaveNonDim as ND
import OneD.NBody as NB
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize

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

def Startup_Choice():
    print("")
    print("Choose a (non-dimensional) box length:")
    L = float(input())
    print("")
    print("Now, Fuzzy DM [1], Particles[2], or both[3]?")
    print("Enter [1/2/3]")
    choice = int(input())
    return L, choice 

def Startup_Initial_Parameters(choice, hbar,L_scale,v_scale,M_scale):
    print("")
    print("INITIAL Parameters:")
    if choice == 1: #FDM
        print("Choose a (non-dimensional) Boson mass:")
        mu = float(input())
        print("How many Bosons?")
        Num_bosons = int(input())

        #Calculate dimensional mass:
        m = mu*M_scale
        print(f"Mass mu = {mu}, m = mu*M = {m}")
        #Calculate Fuzziness:
        r = ND.r(hbar,m,v_scale,L_scale)
        print(f"Fuzziness: r = {r}")

        return mu, Num_bosons, r

    elif choice == 2: #Particles
        print("Choose your particle/star mass (per unit area):")
        sigma = float(input())
        print("How many particles?")
        Num_stars = int(input())
        
        return sigma, Num_stars

    elif choice == 3: #FDM + Particles
        #FDM
        print("Choose a (non-dimensional) Boson mass:")
        mu = float(input())
        print("How many Bosons?")
        Num_bosons = int(input())
        #Calculate dimensional mass:
        m = mu*M_scale
        print(f"Mass mu = {mu}, m = mu*M = {m}")
        #Calculate Fuzziness:
        r = ND.r(hbar,m,v_scale,L_scale)
        print(f"Fuzziness: r = {r}")
        
        #Particles
        print("")
        print("Choose your particle/star mass (per unit area):")
        sigma = float(input())
        print("How many particles?")
        Num_stars = int(input())
        
        return mu, Num_bosons, r, sigma, Num_stars

def gauss(x,b,std):
    return np.exp(-(x-b)**2/(2*std**2))/(np.sqrt(2*np.pi)*std)

#########################################################3
#FOURIER STUFF

def fourier_gradient(phi,length):
    n = len(phi)
    L = length 

    #1. FFT the density (perturbation)
    phi_n = np.fft.rfft(phi,n) #fft for real input
    
    #2. Compute the fourier coefficients of phi
    grad_n = np.array([]) #empty for storage. Will hold the fourier coefficients
    for nn in range(len(phi_n)):
        k = 2*np.pi*nn/L
        val = -1j*phi_n[nn]*k
        grad_n = np.append(grad_n,val)
    
    #3. IFFT back to get Potential
    grad = np.fft.irfft(grad_n,n) #use Phi_n as Fourier Coefficients
    #grad = np.real(grad)
    return grad

#Determine the potential from poisson's equation using fourier method
def fourier_potential(chi,length):
    n = len(chi)
    L = length #length of box

    #1. FFT the norm-squared of the wave-function (minus it's mean background)
    rho = np.absolute(chi)**2
    rho_avg = np.mean(rho)
    p = rho-rho_avg
    p_n = np.fft.rfft(p,n) #fft for real input
    
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
    phi = np.fft.irfft(phi_n,n) #use Phi_n as Fourier Coefficients
    return phi

def fourier_potentialV2(rho_nondim,length):
    rho = rho_nondim
    n = len(rho)
    L = length #length of box

    #1. FFT the norm-squared of the wave-function (minus it's mean background)
    rho_avg = np.mean(rho)
    p = 4*np.pi*(rho-rho_avg)
    p_n = np.fft.rfft(p,n) #fft for real input
    
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
    phi = np.fft.irfft(phi_n,n) #use Phi_n as Fourier Coefficients
    return phi
#####################################################3
#Full Calculation/Simulation Functions

def run_FDM(z, L, dz, mu, Num_bosons, r, v_s, L_s, Directory, folder_name):
    M_s = v_s**2 * L_s
    T = L_s / v_s
    #####################################################
    # INITIAL SETUP
    #####################################################
    #Set an initial wavefunction
    b=0
    #print("Choose the standard deviation of the initial distribution:")
    #std = float(input())
    std=0.1*L
    psi = np.sqrt(gauss(z,b,std)*Num_bosons)#*Num_particles / (L**3))
    chi = psi*L_s**(3/2)

    #Calculate initial Density perturbation (non-dimensionalized and reduced)
    rho = np.absolute(chi)**2 #just norm-squared of wavefunction
    
    #Calculate initial (non-dim) potential
    phi = fourier_potentialV2(rho,L) 

    #Check how it's normalized:
    print(f"|chi|^2 = {np.sum(dz*np.absolute(chi)**2)}")
    print(f"Numerically calculated: |psi|^2 = {np.sum(dz*np.absolute(psi)**2)}")

    m = mu*M_s
    Rho_avg = m*np.mean(np.absolute(psi)**2)
    T_collapse = 1/Rho_avg**0.5
    tau_collapse = T_collapse/T
    print(f"(Non-dim) Collapse time: {tau_collapse}")
    
    #To fix plot axis limits:
    #y0_max = np.max(phi)*1.5
    y0_max = np.max(rho)*10
    y1_max = v_s*100
    eta = 0.025*L #resolution for Husimi
    dtau = 0.01*tau_collapse
    tau_stop = tau_collapse*2 #t_stop/T
    time = 0
    i = 0
    while time <= tau_stop:
        ######################################
        #Evolve system forward by one time-step
        chi,phi,rho = ND.time_evolveV2(chi,phi,r,dz,dtau,m,L)

        #PHASE SPACE CALCULATION:
        k = 2*np.pi*np.fft.fftfreq(len(z),dz)
        #rescale wavenumber k to velocity v:
        hbar = 1
        v = k*(hbar/m)

        x_min, x_max = np.min(z), np.max(z)
        v_min, v_max = np.min(v), np.max(v)
        
        F = ND.Husimi_phase(chi,z,dz,L,eta)
        
        ######################################
        #Plot everything and save the file
        fig,ax = plt.subplots(1,2,figsize = (20,10))
        plt.suptitle("Time $\\tau = $"+ f"{round(dtau*i,5)}".zfill(5), fontsize = 20)    
        
        ax[0].plot(z,chi.real, label = "Re[$\\chi$]")
        ax[0].plot(z,chi.imag, label = "Im[$\\chi$]")
        ax[0].plot(z,phi,label = "Potential [Fourier perturbation]")
        ax[0].plot(z,rho,label = "$\\rho = \\chi \\chi^*$")
        ax[0].set_ylim([-y0_max, y0_max] )
        ax[0].set_xlabel("$z = x/L$")
        ax[0].legend()
        
        if i == 0:
            max_F = np.max(F) #max_F = 0.08
        ax[1].imshow(F,extent = (x_min,x_max,v_min,v_max),cmap = cm.hot, norm = Normalize(0,max_F), aspect = (x_max-x_min)/(2*y1_max))
        ax[1].set_xlim([x_min,x_max])
        ax[1].set_ylim([-y1_max,y1_max]) #[v_min,v_max])
        ax[1].set_xlabel("$z = x/L$")
        #ax[1].colorbar()

        #now save it as a .jpg file:
        folder = Directory + "/" + folder_name
        filename = 'ToyModelPlot' + str(i+1).zfill(4) + '.jpg';
        plt.savefig(folder + "/" + filename)  #save this figure (includes both subplots)
        plt.close() #close plot so it doesn't overlap with the next one

        time += dtau #forward on the clock
        i += 1
    
def run_NBody(z,L,dz,sigma,Num_stars, v_scale, L_scale, Directory):
    M_scale = v_scale**2 * L_scale
    T_scale = L_scale / v_scale
    ########################################################
    # INITIAL SETUP
    ########################################################
    #Set initial distribution on grid
    b = 0 #center at zero
    std = 0.1*L #standard deviation of 1
    z_0 = np.random.normal(b,std,Num_stars) #initial positions sampled from normal distribution
    stars = [NB.star(i,sigma,z_0[i],0) for i in range(len(z_0))] #create list of normally distributed stars, zero initial speed
    
    #reposition stars if they were generated outside the box
    for star in stars:
        if np.absolute(star.x) > L/2:
                star.reposition(L)
    
    folder_name = "SelfGrav_NBody_Images"
    os.chdir(Directory + "/" + folder_name)

    #Calculate distirubtion on Mesh
    grid_counts = NB.grid_count(stars,L,z)
    rho = (grid_counts/dz)*sigma 
        
    #m = mu*M_scale
    Rho_avg = M_scale*np.mean(rho)/L_scale
    T_collapse = 1/(Rho_avg)**0.5
    tau_collapse = T_collapse/T_scale
    print(f"(Non-dim) Collapse time: {tau_collapse}")
    
    dtau = 0.01*tau_collapse
    tau_stop = tau_collapse*2 #t_stop/T
    time = 0
    i = 0 #counter, for saving images
    while time <= tau_stop:
        #################################################
        #CALCULATION OF PHYSICAL QUANTITIES
        #################################################
        #Calculate distirubtion on Mesh
        grid_counts = NB.grid_count(stars,L,z)
        rho = (grid_counts/dz)*sigma 
        
        #Calculate potential 
        phi = fourier_potentialV2(rho,L)
        
        #Calculate Acceleration Field on Mesh:
        a_grid = -NB.acceleration(phi,L) 
        
        #################################################
        # PLOTTING
        #################################################
        fig,ax = plt.subplots(1,3)#3)
        fig.set_size_inches(30,10)
        plt.suptitle("Time $\\tau = $" +f"{round(dtau*i,5)}".zfill(5))
        
        ax[0].plot(z,phi,label = "Potential")
        ax[0].plot(z,rho,label = "Number density")
        ax[0].plot(z,a_grid)
        ax[0].set_xlim([-L/2,L/2])
        ax[0].set_ylim([-0.1*Num_stars/dz,0.1*Num_stars/dz])
        
        #Plot the Phase Space distribution
        x_s = np.array([star.x for star in stars])
        v_s = np.array([star.v for star in stars])
        ax[1].plot(x_s,v_s,'.',label = "Phase Space Distribution")
        ax[1].set_ylim([-v_scale*50,v_scale*50])
        ax[1].set_xlim([-L/2,L/2])
        ax[1].legend()

        #Plot Phase space distribution in another way
        heat = ax[2].hist2d(x_s,v_s,bins = [200,200],range = [[-L/2,L/2],[-2,2]],cmap = cm.hot)
        #ax[2].set_colorbar()
        ax[2].set_xlim(-L/2,L/2)
        ax[2].set_ylim(-15,15)
        #fig.colorbar(heat[3], ax[2])

        #ADDITIONAL:
        #PLOT CENTER OF MASS
        centroid_z = 0
        for j in range(len(grid_counts)):
            centroid_z += z[j]*grid_counts[j]
        centroid_z = centroid_z / Num_stars
        ax[1].scatter(centroid_z,0,s = 100,c = "r",marker = "o")
        
        #now save it as a .jpg file:
        folder = Directory + "/" + folder_name
        filename = 'ToyModelPlot' + str(i+1).zfill(4) + '.jpg';
        plt.savefig(folder + "/" + filename)  #save this figure (includes both subplots)
        plt.close() #close plot so it doesn't overlap with the next one
        
        ############################################################
        #EVOLVE SYSTEM (After calculations on the Mesh)
        ############################################################
        #1,2: Kick+Drift
        g = NB.accel_funct(a_grid,L,dz)

        for star in stars:
            #print(star.x)
            star.kick_star(g,dtau/2)
            star.drift_star(dtau)

            #corrective maneuvers on star position
            #(for positions that drift outside of the box...
            # must apply periodicity)
            if np.absolute(star.x) > L/2:
                print(f"z = {star.x}")
                modulo = (star.x // (L/2))
                remainder = star.x % (L/2)
                print(f"mod = {modulo}, remainder = {remainder}")
                if modulo % 2 == 0: #check if modulo is even
                    star.x = remainder 
                else: #if modulo is odd, further check:
                    if star.x > 0:
                        star.x = remainder-L/2
                    elif star.x < 0:
                        star.x = remainder+L/2
                print(f"new z = {star.x}")
                print(" ")
        #3,4: Re-update potential and acceleration fields, + Kick
        grid_counts = NB.grid_count(stars,L,z)
        rho = (grid_counts/dz)*sigma 
        phi = fourier_potentialV2(rho,L)
        a_grid = -NB.acceleration(phi,L) 
        g = NB.accel_funct(a_grid,L,dz)
        for star in stars:
            star.kick_star(g,dtau/2)
            
        time += dtau
        i += 1

def run_FDM_n_Bodies(sim_choice, z, L, dz, mu, Num_bosons, r, sigma, Num_stars, v_s, L_s, Directory, folder_name):
    M_s = v_s**2 * L_s
    T_s = L_s / v_s

    ########################################################
    # INITIAL SETUP
    ########################################################
    # FOR THE FDM
    #Set an initial wavefunction
    b=0
    print("Choose the standard deviation of the initial FDM distribution (as a fraction of the box width):")
    std = float(input())
    std=std*L
    psi = np.sqrt(gauss(z,b,std)*Num_bosons)#*Num_particles / (L**3))
    chi = psi*L_s**(3/2)

    #Calculate initial Density perturbation (non-dimensionalized and reduced)
    rho = np.absolute(chi)**2 #just norm-squared of wavefunction
    
    #Calculate initial (non-dim) potential
    phi = fourier_potentialV2(rho,L) 

    #Check how it's normalized:
    print(f"|chi|^2 = {np.sum(dz*np.absolute(chi)**2)}")
    print(f"Numerically calculated: |psi|^2 = {np.sum(dz*np.absolute(psi)**2)}")

    m = mu*M_s

    ####################################################
    # FOR THE PARTICLES
    #Set initial distribution on grid
    b = 0 #center at zero
    print("Choose the standard deviation of the Particle system (as fraction of the box width):")
    std = float(input())
    std = std*L 
    z_0 = np.random.normal(b,std,Num_stars) #initial positions sampled from normal distribution
    stars = [NB.star(i,sigma,z_0[i],0) for i in range(len(z_0))] #create list of normally distributed stars, zero initial speed
    
    #reposition stars if they were generated outside the box
    for star in stars:
        if np.absolute(star.x) > L/2:
                star.reposition(L)

    #Calculate distribution on Mesh
    grid_counts = NB.grid_count(stars,L,z)
    rho = (grid_counts/dz)*sigma 
    rho += np.absolute(chi)**2
    
    ##########################################################
    #PLOT AXIS LIMITS:
    #y0_max = np.max(phi)*1.5
    y00_max = np.max((grid_counts/dz)*sigma)*10
    y01_max = v_s*200
    y10_max = np.max(np.absolute(chi)**2)*10
    y11_max = y01_max #v_s*100
    ###################################################
    #PHASE SPACE STUFF
    eta = 0.05*L*r #resolution for Husimi
    k = 2*np.pi*np.fft.fftfreq(len(z),dz)
    #rescale wavenumber k to velocity v:
    hbar = 1
    v = k*(hbar/m)
    x_min, x_max = np.min(z), np.max(z)
    v_min, v_max = np.min(v), np.max(v)
    print(f"v_min = {v_min}, v_max = {v_max}")
    #v_max = np.max(np.abs([np.min(v),np.max(v)]))
    #v_min = -v_max
    #print(f"v_min = {v_min}, v_max = {v_max}")
    if v_max >= np.abs(v_min):
        v_min = -v_max 
    elif np.abs(v_min) > v_max:
        v_max = -v_min
    print(f"v_min = {v_min}, v_max = {v_max}")
    
    ####################################################
    #PRE-LOOP TIME-SCALE SETUP
    #m = mu*M_scale
    Rho_avg = M_s*np.mean(rho)/L_s
    print(Rho_avg)
    T_collapse = 1/(Rho_avg)**0.5
    tau_collapse = T_collapse/T_s
    print(f"(Non-dim) Collapse time: {tau_collapse}")

    dtau = 0.01*tau_collapse
    if sim_choice == 1:
        tau_stop = tau_collapse*2 #t_stop/T
    elif sim_choice == 2:
        tau_stop = tau_collapse*64
        collapse_index = int(np.floor(tau_collapse/dtau))
        snapshot_indices = np.multiply(collapse_index-1,[0,1,2,4,8,16,32,64])
        print(f"Snapshots at i = {snapshot_indices}")
    time = 0
    i = 0 #counter, for saving images
    os.chdir(Directory + "/" + folder_name) #Change Directory to where Image Folders are
    while time <= tau_stop:
        overflow = checkMemory(mem_limit = 75)
        if overflow == True:
            break
        #################################################
        #CALCULATION OF PHYSICAL QUANTITIES
        #################################################
        #PHASE SPACE CALCULATION:
        F = ND.Husimi_phase(chi,z,dz,L,eta)

        #POSITION SPACE CALCULATIONS:
        #1. Calculate the Total Density

        #Calculate Particle distirubtion on Mesh
        grid_counts = NB.grid_count(stars,L,z)
        rho = (grid_counts/dz)*sigma 

        #Add the density from the FDM
        rho += np.absolute(chi)**2
        
        #2. Calculate potential 
        phi = fourier_potentialV2(rho,L)

        #3. Calculate Acceleration Field on Mesh:
        a_grid = -NB.acceleration(phi,L) 
        
        #################################################
        # PLOTTING
        # Plot everytime if sim_choice == 1
        # Plot only specific time steps if sim_choice == 2
        #################################################
        
        if sim_choice == 1:
            PLOT = True #always plot
        elif sim_choice == 2:
            #check if time-step is correct one.
            if i in snapshot_indices:
                PLOT = True
            else:
                PLOT = False
                
        if PLOT == True:
            layout = [['upper left', 'upper right', 'far right'],
                    ['lower left', 'lower right', 'far right']]

            fig, ax = plt.subplot_mosaic(layout, constrained_layout = True)
            fig.set_size_inches(20,10)
            plt.suptitle("Time $\\tau = $" +f"{round(dtau*i,5)}".zfill(5), fontsize = 20)    
            
            ##############################################3
            #ACCELERATIONS
            body_accel = -NB.acceleration(fourier_potentialV2((grid_counts/dz)*sigma,L),L)
            if i == 0: #want to set a limit on the acceleration graph
                a = np.abs([np.max(body_accel),np.min(body_accel)])
                y_lim_max = np.max(a)*2
            ax['far right'].plot(z, body_accel, label = "Particle Contribution")
            ax['far right'].plot(z, fourier_gradient(fourier_potentialV2(np.abs(chi)**2,L),L), label = "FDM Contribution")
            ax['far right'].set_ylim(-y_lim_max,y_lim_max)
            ax['far right'].set_title("Acceleration contributions",fontsize = 15)
            ax['far right'].legend(fontsize = 20)
            
            # THE FDM
            #ax['upper left'].plot(z,chi.real, label = "Re[$\\chi$]")
            #ax['upper left'].plot(z,chi.imag, label = "Im[$\\chi$]")
            ax['upper left'].plot(z,phi,label = "$\\Phi_{FDM}$ [Fourier perturbation]")
            ax['upper left'].plot(z,np.abs(chi)**2,label = "$\\rho_{FDM} = \\chi \\chi^*$")
            ax['upper left'].set_ylim([-y00_max, y00_max] )
            ax['upper left'].set_xlabel("$z = x/L$")
            ax['upper left'].legend(fontsize = 15)
            ax['upper left'].set_title("Non-Dimensional Densities and Potentials",fontsize = 15)
            
            if i == 0:
                max_F = np.max(F)/2
            ax['upper right'].imshow(F,extent = (x_min,x_max,v_min,v_max),cmap = cm.hot, norm = Normalize(0,max_F), aspect = (x_max-x_min)/(2*y01_max))
            ax['upper right'].set_xlim(x_min,x_max)
            ax['upper right'].set_ylim(-y01_max,y01_max) #[v_min,v_max])
            ax['upper right'].set_xlabel("$z = x/L$")
            ax['upper right'].set_title("Phase Space Distributions", fontsize = 15)
            
            ##############################################3
            # THE PARTICLES
            ax['lower left'].plot(z,phi,label = "$\\Phi_{Particles}$ [Fourier perturbation]")
            ax['lower left'].plot(z,(grid_counts/dz)*sigma,label = "$\\rho_{Particles}$")
            #ax['lower left'].set_xlim(-L/2,L/2)
            ax['lower left'].set_ylim(-y10_max,y10_max)
            ax['lower left'].legend(fontsize = 15)

            #Plot the Phase Space distribution
            x_s = np.array([star.x for star in stars])
            v_s = np.array([star.v for star in stars])
            ax['lower right'].scatter(x_s,v_s,s = 1,label = "Particles")
            ax['lower right'].set_ylim(-y11_max,y11_max)
            ax['lower right'].set_xlim(-L/2,L/2)
            ax['lower right'].legend(fontsize = 15)

            #ADDITIONAL:
            #PLOT CENTER OF MASS
            centroid_z = 0
            for j in range(len(grid_counts)):
                centroid_z += z[j]*grid_counts[j]
            centroid_z = centroid_z / Num_stars
            ax['lower right'].scatter(centroid_z,0,s = 100,c = "r",marker = "o")
            
            #now save it as a .jpg file:
            folder = Directory + "/" + folder_name
            filename = 'ToyModelPlot' + str(i+1).zfill(4) + '.jpg';
            plt.savefig(folder + "/" + filename)  #save this figure (includes both subplots)
            plt.clf()
            plt.cla()
            plt.close(fig) #close plot so it doesn't overlap with the next one

        ############################################################
        #EVOLVE SYSTEM (After calculations on the Mesh)
        ############################################################
        #1,2: Kick+Drift

        #FUZZY DM
        #1. Kick in current potential
        chi= ND.kick(chi,phi/2,r,dtau/2)
        #2. Drift in differential operator
        chi = ND.drift(chi,r,dz,dtau)

        #PARTICLES
        g = NB.accel_funct(a_grid,L,dz)
        for star in stars:
            #print(star.x)
            star.kick_star(g,dtau/2)
            star.drift_star(dtau)

            #corrective maneuvers on star position
            #(for positions that drift outside of the box...
            # must apply periodicity)
            if np.absolute(star.x) > L/2:
                star.reposition(L)

        #3 Re-update potential and acceleration fields
        
        #Calculate Particle distribution on Mesh
        grid_counts = NB.grid_count(stars,L,z)
        rho = (grid_counts/dz)*sigma 
        #Add the density from the FDM
        rho += np.absolute(chi)**2
        #Calculate potential 
        phi = fourier_potentialV2(rho,L)
        #Calculate Acceleration Field on Mesh:
        a_grid = -NB.acceleration(phi,L) 

        #4. KICK in updated potential

        #FUZZY DM
        chi = ND.kick(chi,phi/2,r,dtau/2)

        #PARTICLES
        a_grid = -NB.acceleration(phi,L) 
        g = NB.accel_funct(a_grid,L,dz)
        for star in stars:
            star.kick_star(g,dtau/2)
            
        #Note: no need to re-calculate density.
        # That happens again at start of loop
        time += dtau
        i += 1

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
            imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS) 
            imResize.save( file, 'JPEG', quality = 1080) # setting quality
            # printing each resized image name
            #print(im.filename.split('\\')[-1], " is resized") 
    
    # Calling the generate_video function
    generate_video(fourcc,Directory, folder_name, video_name,dt)
