import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
import os
import subprocess
import cv2 
from PIL import Image 
import scipy.optimize as opt
from scipy.stats import gaussian_kde
from scipy.special import erf as erf

import OneD.FDM as FDM_OG
import OneD.NBody as NB
import OneD.Global as GF

import OneD.Analysis.FDM as FDM
import OneD.Analysis.NBody as NBody

#Set up Directory for saving files/images/videos
# Will not rename this again
dirExtension = "1D_Codes/Non-Dim/Analysis"
Directory = os.getcwd()#+"/"+dirExtension #os.curdir() #"/home/boris/Documents/Research/Coding/1D codes/Non-Dim"
print(Directory)

######################################################
# MY ONE BIG FUNCTION FOR ALL ANALYSIS:
######################################################
def analysis(folder: str, type = 'Periodic'):#,*args):
    os.chdir(folder)
    print(os.getcwd())

    #Load in the necessary quantities:
    Properties = np.loadtxt("Properties.csv", dtype = str, delimiter = ",")
    # print(Properties)
    Properties = [Properties[:,1][i] for i in range(len(Properties))]
    Time_elapsed, L, mu,Num_bosons = [float(Properties[i]) for i in range(4)]
    fraction, Num_stars,r,N = [float(Properties[i]) for i in range(5,len(Properties)-2)]
    variable_mass = Properties[4]
    variable_mass = [variable_mass, fraction]
    indices = Properties[-2].replace('[','').replace(']','')
    indices = indices.split(' ')
    new_indices = []
    for x in indices:
        try: 
            value = int(x)
            new_indices.append(value)
        except:
            pass    
    indices = new_indices
    # print(indices)

    dtau = float(Properties[-1])
    print(f"r={r},Num_stars = {Num_stars}")

    time = dtau*np.arange(indices[-1]+1)
        
    # percent_FDM = Num_bosons*mu / (Num_bosons*mu + Num_stars*sigma)

    N = int(N)
    z = np.linspace(-L/2,L/2,N)
    dz = z[1]-z[0]

    # print("Periodic [1] or Isolated [2] BC's?")
    # bc = int(input())
    
    if type == 'Periodic':
        #type = 'Periodic'
        G_tilde = None
    elif type == 'Isolated':
        #type = 'Isolated'
        z_long = np.linspace(-L,L,2*N-1)
        G = np.abs(z_long)
        G_tilde = np.fft.fft(G)

    if Num_bosons == 0:
        stars_x = np.loadtxt("StarsOnly_Pos.csv", dtype = float, delimiter=",")
        stars_v = np.loadtxt("StarsOnly_Vel.csv", dtype = float, delimiter=",")
        stars_m = np.loadtxt("Particle_masses.csv", dtype = float, delimiter=",")
        K_Energies = np.loadtxt("K_star_Energies.csv", dtype = float,delimiter = ",")
        W_Energies = np.loadtxt("W_star_Energies.csv", dtype = float,delimiter = ",")
        K_fine_Energies = np.loadtxt("K_star_fine_Energies.csv", dtype = float,delimiter = ",")
        W_fine_Energies = np.loadtxt("W_star_fine_Energies.csv", dtype = float,delimiter = ",")
        W2_Energies = np.loadtxt("W_2_star_Energies.csv",dtype=float,delimiter=",")
        
        chi = np.loadtxt(f"Chi.csv", dtype = complex, delimiter=",")
        part_centroids = np.loadtxt("Particle_Centroids.csv",dtype = float, delimiter=',')
        #z_rms_storage = None#np.loadtxt("z_rms_storage.csv", dtype = float, delimiter=",")
        #v_rms_storage = None#np.loadtxt("v_rms_storage.csv", dtype = float, delimiter=",")
        z_rms_storage = np.loadtxt("z_rms_storage.csv", dtype = float, delimiter=",")
        v_rms_storage = np.loadtxt("v_rms_storage.csv", dtype = float, delimiter=",")
                
    elif Num_stars == 0:
        chi = np.loadtxt("Chi.csv", dtype = complex, delimiter=",")
        Ks_FDM = np.loadtxt("K_FDM_storage.csv",dtype=float,delimiter=",")
        Ws_FDM = np.loadtxt("W_FDM_storage.csv",dtype=float,delimiter=",")
        fdm_centroids = np.loadtxt("FDM_Centroids.csv",dtype = float, delimiter=',')
        stars_x = None
        star_v = None
        K_Energies = None 
        W_Energies = None

    elif Num_bosons!=0 and Num_stars !=0:
        stars_x = np.loadtxt("StarsOnly_Pos.csv", dtype = float, delimiter=",")
        stars_v = np.loadtxt("StarsOnly_Vel.csv", dtype = float, delimiter=",")
        stars_m = np.loadtxt("Particle_masses.csv", dtype = float, delimiter=",")
        chi = np.loadtxt("Chi.csv", dtype = complex, delimiter=",")
        part_centroids = np.loadtxt("Particle_Centroids.csv",dtype = float, delimiter=',')
        fdm_centroids = np.loadtxt("FDM_Centroids.csv",dtype = float, delimiter=',')
        z_rms_storage = np.loadtxt("z_rms_storage.csv", dtype = float, delimiter=",")
        v_rms_storage = np.loadtxt("v_rms_storage.csv", dtype = float, delimiter=",")
        
        K_Energies = np.loadtxt("K_star_Energies.csv", dtype = float,delimiter = ",")
        W_Energies = np.loadtxt("W_star_Energies.csv", dtype = float,delimiter = ",")
        K_fine_Energies = np.loadtxt("K_star_fine_Energies.csv", dtype = float,delimiter = ",")
        W_fine_Energies = np.loadtxt("W_star_fine_Energies.csv", dtype = float,delimiter = ",")
        W2_Energies = np.loadtxt("W_2_star_Energies.csv",dtype=float,delimiter=",")
        
        Ks_FDM = np.loadtxt("K_FDM_storage.csv",dtype=float,delimiter=",")
        Ws_FDM = np.loadtxt("W_FDM_storage.csv",dtype=float,delimiter=",")
    
    do_spectra = False
    if Num_bosons!=0 and do_spectra==True:
        #Get Spitzer Density:
        E0,sigma,f0 = .7, .5, .1
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
        #initial conditions:
        y = np.zeros(2) #phi = 0, dphi/dt = 0 

        #z_original = np.copy(z)
        dz = z[1]-z[0]
        #N = len(z)
        # z = np.zeros(len(z)//2)
        z_trunc = 0
        phi = np.zeros(len(z)//2)
        rho = np.zeros(len(z)//2)
        rho[0] = density(phi[0])
        i = 0
        while rho[i] > 0 and i<len(z)//2-1:
            k1 = derivs(y)
            k2 = derivs(y + dz*k1/2.)
            k3 = derivs(y + dz*k2/2.)
            k4 = derivs(y + dz*k3)
            y = y + (k1 + 2.*k2 + 2.*k3 + k4)*dz/6.
            
            i+=1
            phi[i] = y[0]
            rho[i] = density(y[0])
            z_trunc+=dz

        imax = i    
        rho = rho[:imax]
        rho = np.append(rho,np.zeros(len(z)//2 - len(rho)))
        rho = np.append(rho[::-1],rho)

        chi_storage = np.loadtxt("chi_storage.csv", dtype=complex, delimiter=",")
        rho_FDM_storage = mu*np.abs(chi_storage)**2
        if len(rho_FDM_storage[0])>len(rho):
            rho = np.append(rho,[0])
        delta_FDM = np.array([rho_FDM_storage[i] - rho for i in range(len(rho_FDM_storage))])
        N_t = np.shape(rho_FDM_storage)[0]

        # z_ij,t_ij = np.meshgrid(z,time)

        # plt.figure()
        # plt.title("FDM Density over position and time: $\\rho(x,t)$ ")
        # plt.pcolormesh(z_ij, t_ij, rho_FDM_storage, cmap = "coolwarm")
        # plt.xlabel("Position $x$")
        # plt.ylabel("Time $t$")
        # plt.colorbar()
        # plt.show()

        spectrum = np.fft.fft2(delta_FDM)
        spectrum = np.fft.fftshift(spectrum)
        power_spectrum = np.abs(spectrum)**2
        k = 2*np.pi*np.fft.fftfreq(N, d=dz)
        w = 2*np.pi*np.fft.fftfreq(N_t, d=dtau)
        
        k = np.fft.fftshift(k)
        w = np.fft.fftshift(w)
        k_ij,w_ij = np.meshgrid(k,w)

        plt.figure()
        plt.title("Power Spectrum of Density Fluctuations: $\\hat{\\rho}(k,\\omega)$")
        pcm = plt.pcolormesh(k_ij, w_ij, power_spectrum, norm = LogNorm(), cmap = "coolwarm")
        plt.xlabel("Wavenumer $k = 2\\pi/\\lambda$")
        plt.ylabel("Frequency $\\omega$")
        plt.colorbar(pcm)
        plt.show()
        
        # fig,ax = plt.subplots(1,2)
        # ax[0].plot(z,delta_FDM[0,:])
        # ax[1].plot(1/np.fft.rfftfreq(N,d=dz)[1:],np.abs(np.fft.rfft(delta_FDM[0,:])[1:])**2)
        # ax[1].set_yscale("log")
        # # ax[1].set_xlim(0,50)
        # plt.show()

        # fig,ax = plt.subplots(1,2)
        # ax[0].plot(time,delta_FDM[:,N//2])
        # ax[1].plot(2*np.pi*np.fft.rfftfreq(N_t,d=dtau)[1:],np.abs(np.fft.rfft(delta_FDM[:,N//2])[1:])**2)
        # ax[1].set_yscale("log")
        # # ax[1].set_xlim(0,20)
        # plt.show()

    #Setup our data post-import:
    #print(stars_x)
    if stars_x is not None:
        if variable_mass[0] == True:
            num_to_change = int(np.floor(fraction*Num_stars))
            
            masses1 = stars_m[0]
            masses2 = stars_m[1]
            stars1 = NB.stars(masses1,[stars_x[i] for i in range(0,num_to_change)],[stars_v[i] for i in range(0,num_to_change)])#[NB.star(i,sigma1,stars_x[i],stars_v[i]) for i in range(0,num_to_change)] 
            stars2 = NB.stars(masses2,[stars_x[i] for i in range(num_to_change,Num_stars)],[stars_v[i] for i in range(num_to_change,Num_stars)]) 
            stars = [stars1[:],stars2[:]]
            
        else:
            stars = NB.stars(stars_m,[stars_x[i] for i in range(len(stars_x))],[stars_v[i] for i in range(len(stars_v))])
    else: 
        stars = []
    m=mu #M_s = 1
    # if variable_mass[0]=='False':
    #     #rho_part, rho_FDM = plot_FDMnBodies(z,L,m,mu,sigma,r,stars,chi,type,G_tilde)
    # else:
    #     rho_part = NB.particle_density(stars,L,z,variable_mass)
    #     rho_FDM = np.zeros_like(rho_part)
    # phi_part = GF.fourier_potential(rho_part,L,type = type, G_tilde=G_tilde)
    # phi_FDM = GF.fourier_potential(rho_FDM,L,type = type, G_tilde=G_tilde)
    # phi = phi_part + phi_FDM
    
    #######################################################
    ### FDM ANALAYSIS #######################################
    if Num_bosons != 0:
        
        #NBody.plot_centroids(indices,fdm_centroids)
        
        # time = dtau*np.linspace(0,len(Ks_FDM),len(Ks_FDM))
        if Num_stars==0:
            RMS_amplitude, Max_amplitude = FDM.plot_Energies(time,Ks_FDM,Ws_FDM)

        #     FDM.plot_Freqs(time, Ks_FDM,Ws_FDM)

        v_dist = FDM.v_distribution(z,L,chi,r,mu)

        FDM_z_rms,FDM_v_rms = FDM.rms_stuff(z,L,chi,v_dist,mu)
        print(f"z_rms = {FDM_z_rms}")
        print(f"v_rms = {FDM_v_rms}")

    #######################################################
    ### NBODY ANALYSIS ######################################
    if Num_stars != 0:
        #NBody.plot_centroids(indices,part_centroids)
        

        #Calculate rms velocity and position
        #z_rms,v_rms = NBody.rms_stuff(sigma,stars,phi_part,L,z,dz,type = type)

        #z_rms_storage =np.append(z_rms_storage,z_rms)
        #v_rms_storage =np.append(v_rms_storage,v_rms)
        #NBody.rms_plots(indices,z_rms_storage,v_rms_storage)

        #NBody.v_distribution(stars,L)

        #NBody.select_stars_plots(z,K_5stars_Energies,W_5stars_Energies)
        #dtau = 0.5*0.004831915000023168
        print(W2_Energies.shape)
        print(K_fine_Energies.shape)
        
        if variable_mass[0]=='True' or Num_bosons!=0:
            W_fine_Energies = W2_Energies

        time = dtau*np.linspace(0,len(K_fine_Energies),len(K_fine_Energies))
        if Num_bosons==0:
            RMS_amplitude, Max_amplitude = NBody.plot_Energies(time,K_fine_Energies,W_fine_Energies,variable_mass)
        
        if variable_mass[0]=='False' and Num_bosons==0:
            NBody.plot_Freqs(time, K_fine_Energies,W_fine_Energies)

        #NBody.all_stars_plots(time,K_Energies,W2_Energies, variable_mass=variable_mass)
        #NBody.all_stars_plots(np.linspace(0,2.47943,len(K_fine_Energies[:,0])), K_fine_Energies,W_fine_Energies, variable_mass=variable_mass)

        #Correct the potential's minimum point drift:
        if variable_mass[0] == 'True':
            fraction = variable_mass[1]
            Num_stars = len(W_Energies[0])
            num_to_change = int(np.floor(fraction*Num_stars))
            
            for i in range(len(W_Energies)):
                j = indices[i]
                correction = (2/Num_stars)*(0.5*np.sum(W_Energies[i])-(W2_Energies[j,0]+W2_Energies[j,1]))
                W_Energies[i] = W_Energies[i] - correction

            #check correction:
            print(W2_Energies[-1,0]+W2_Energies[-1,1])
            print(0.5*np.sum(W_Energies[-1]))

        else:
            for i in range(len(W_Energies)):
                j = indices[i]
                correction = (2/len(W_Energies[i]))*(0.5*np.sum(W_Energies[i]) - W2_Energies[-1])
                W_Energies[i] = W_Energies[i] - correction

            #check correction:
            print(W2_Energies[-1])
            print(0.5*np.sum(W_Energies[-1]))

        Energies = W_Energies + K_Energies
        Energies_i = Energies[0]
        Energies_f = Energies[-1]

        
        # if variable_mass[0]=='True' or Num_bosons!=0:
        #     deltaE = NBody.scatter_deltaE(Energies_i, Energies_f, variable_mass, Num_bosons, r)
        #     deltaE = NBody.scatter_deltaE_frac(Energies_i, Energies_f, variable_mass, Num_bosons, r)
        deltaE=0

        deltaE_array = np.array([])
        for i in range(len(indices)):
            value = np.mean((Energies[i] - Energies_i)/Energies_i)
            deltaE_array = np.append(deltaE_array,value)
        print(deltaE_array)

        # fig, ax = plt.subplots(1,3)
        # ax[0].plot([np.sum(K_fine_Energies[i,:]) for i in range(1000)],label ="Kinetic")
        # ax[1].plot([np.sum(W_fine_Energies[i,:]) for i in range(1000)],label ="Potential")
        # ax[2].plot([np.sum(K_fine_Energies[i,:])+np.sum(W_fine_Energies[i,:]) for i in range(1000)],label ="K+W")
        
        # plt.legend()
        # plt.show()
        #NBody.all_stars_plots(np.arange(1000,4999),K_fine_Energies,W_fine_Energies)
        
        # plt.figure()
        # plt.scatter([star.x for star in stars],[star.get_W(z,phi,L) for star in stars])
        # plt.show()
        # phi_part = GF.fourier_potential(rho_part,L,type = 'Isolated', G_tilde = G_tilde)
        # print(f"Total Potential: {np.sum(phi_part*rho_part)*dz}")
        # print(f"Total Potential: {np.sum(-1*z*np.gradient(phi_part,dz)*rho_part)*dz}")
        
        
        # #Hist the final energies:
        # Energies = K_Energies+W_Energies
        # n_bins = int(np.floor(np.sqrt(Num_stars)))#int(np.floor(np.sqrt(len(Energies))))
        # plt.hist(Energies[-1,:],bins = n_bins)#100)
        # plt.title("Histogram of Final Energies of Stars")
        # plt.xlabel("Energy (code units)")
        # plt.show()
        
        if type == 'Periodic':
            print("Curve Fitting Procedure ... ")
            NBz_whole,NBrho_whole = NBody.rho_distribution(z,rho_part)
            popt = curve_fitting(L,NBz_whole,NBrho_whole,type = type, G_tilde  = G_tilde)
            print(f"fit params = {popt}")
        else:
            popt = None

    if Num_bosons != 0 and Num_stars != 0:
        
        fig, ax = plt.subplots(2,2, figsize = (20,10), sharex=True, gridspec_kw = {'height_ratios': [2.5,1]})

        V1 = Ks_FDM/np.abs(Ws_FDM)
        V2 = K_fine_Energies/np.abs(W_fine_Energies)
        
        K1 = (Ks_FDM - Ks_FDM[0])/Ks_FDM[0] 
        W1 = (Ws_FDM - Ws_FDM[0])/Ws_FDM[0] 
        E1 = Ks_FDM+Ws_FDM
        E1 = (E1 - E1[0])/E1[0] 

        K2 = (K_fine_Energies - K_fine_Energies[0])/K_fine_Energies[0]
        W2 = (W_fine_Energies - W_fine_Energies[0])/W_fine_Energies[0]
        E2 = K_fine_Energies+W_fine_Energies
        E2 = (E2 - E2[0])/E2[0]
        

        plt.suptitle("Fractional Change in Total Energies in Particles over Time", fontsize = 15) #fontdict={'fontsize' : 15})
        
        ax[0,0].plot(time,K1,label = "$\\Delta K / K_0$ Kinetic Energy")
        ax[0,0].plot(time,W1,label = "$\\Delta W / W_0$ Potential Energy")
        ax[0,0].plot(time,E1,label = "$\\Delta (K+W) / (K_0+W_0)$ Total Energy")
        ax[0,0].set_title("FDM")

        ax[0,1].plot(time,K2,label = "$$\\Delta K / K_0$ Kinetic Energy")
        ax[0,1].plot(time,W2,label = "$\\Delta W / W_0$ Potential Energy")
        ax[0,1].plot(time,E2,label = "$\\delta (K+W) / (K_0+W_0)$ Total Energy")
        ax[0,1].set_title("Particles")
        
        ax[0,0].legend(loc='upper right')
        ax[0,1].legend(loc='upper right')

        ax[1,0].set_title("FDM Ratio of Kinetic to Potential Energy $\\frac{K}{|W|}$", fontdict={'fontsize' : 15})
        ax[1,0].plot(time,V1)
        ax[1,1].set_title("Particles Ratio of Kinetic to Potential Energy $\\frac{K}{|W|}$", fontdict={'fontsize' : 15})
        ax[1,1].plot(time,V2)
        

        ax[0,0].set_ylabel("$\\frac{\\Delta E}{E_0}$")
        ax[0,0].set_xlabel("Time (code units)")
        ax[0,1].set_xlabel("Time (code units)")
        ax[1,0].set_xlabel("Time (code units)")
        ax[1,1].set_xlabel("Time (code units)")

        ax[0,0].grid(True)
        ax[0,1].grid(True)
        ax[1,0].grid(True)
        ax[1,1].grid(True)
    
        fig.subplots_adjust(hspace = 0.3)
        plt.show()


        #Check that total energy is conserved:
        K = K_fine_Energies + Ks_FDM
        W = W_fine_Energies + Ws_FDM
        Virial_ratios = np.abs(K/W)
    
        fig,ax = plt.subplots(1,4,figsize = (20,5))
        plt.suptitle("Energy of FDM+Stars, at Snapshot times/indices", fontsize = 20)
        ax[0].set_title("Potential Energy over time", fontsize = 15)
        ax[0].plot(time,W,"--",label = "$\\Sigma W$")
        
        ax[1].set_title("Kinetic Energy over time", fontsize = 15)
        ax[1].plot(time,K,"--",label = "$\\Sigma K$")
        ax[1].legend()

        #set the scale:
        Dy = np.max(K)-np.min(K)
        y_min = np.min(K+W)
        y_min = y_min - Dy/2
        y_max = Dy + y_min
        ax[2].set_title("Total Energy K+W over time", fontsize = 15)
        ax[2].plot(time,K+W,"--",label = "$\\Sigma E$")
        ax[2].set_ylim(y_min,y_max)
        ax[2].legend()

        ax[3].set_title("Virial Ratio $|K/W|$ over time", fontsize = 15)
        ax[3].plot(time, Virial_ratios, "b--", marker = "o")
        plt.show()

        fig,ax = plt.subplots(1,2, figsize = (10,5))
        star_total_E = K_fine_Energies + W_fine_Energies
        fdm_total_E = Ks_FDM + Ws_FDM
        E_total = star_total_E + fdm_total_E
        ax[0].plot(time,np.abs(star_total_E/E_total))
        ax[0].set_title("Stars: $E/E_{total}$")
        
        ax[1].plot(time,np.abs(fdm_total_E/E_total))
        ax[1].set_title("FDM: $E/E_{total}$")
        plt.show()

    
    if Num_stars != 0 and Num_bosons != 0:#'FDM_z_rms' in locals() and 'FDM_v_rms' in locals() and 'z_rms' in locals() and 'v_rms' in locals():
        return r, deltaE, deltaE_array, Num_stars, popt
    elif Num_stars != 0 and variable_mass[0] == 'False': 
        return Num_stars, RMS_amplitude, Max_amplitude #, popt
    elif Num_stars != 0 and variable_mass[0] == 'True':
        return fraction, deltaE, deltaE_array
    elif Num_bosons != 0:#'FDM_z_rms' in locals() and 'FDM_v_rms' in locals():
        return r, Num_bosons, FDM_z_rms, FDM_v_rms, RMS_amplitude, Max_amplitude


########################################################
# ALL THE ACTUAL FUNCTIONS:
########################################################
def plot_FDMnBodies(z,L,m,mu,sigma,r,stars,chi,type = 'Periodic', G_tilde = None):
    #stars_x = stars.x
    #stars_v = stars.v
    stars_x = np.zeros(1)
    stars_v = np.zeros(1)
    stars = NB.stars(1,0,0)
    
    Num_stars = len(stars_x)

    #rescale wavenumber k to velocity v:
    dz = z[1]-z[0]
    k = 2*np.pi*np.fft.fftfreq(len(z),dz)
    hbar = 1
    v = k*(hbar/mu)
    x_min, x_max = np.min(z), np.max(z)
    v_min, v_max = np.min(v), np.max(v)


    #Calculate Particle distribution on Mesh
    if Num_stars !=0:
        #stars = [NB.star(i,sigma,stars_x[i],stars_v[i]) for i in range(len(stars_x))]
        grid_counts = NB.grid_count(stars,L,z)
    else:
        grid_counts = np.zeros_like(z)
    rho_part = (grid_counts/dz)*sigma 
    
    #Add the density from the FDM
    rho_FDM = mu*np.absolute(chi)**2 
    rho = rho_FDM + rho_part

    layout = [['upper left', 'upper right', 'far right'],
            ['lower left', 'lower right', 'far right']]

    fig, ax = plt.subplot_mosaic(layout, constrained_layout = True)
    fig.set_size_inches(20,10)
    #plt.suptitle("Time $\\tau = $" +f"{round(dtau*i,5)}".zfill(5), fontsize = 20)    


    ##############################################
    #ADDITIONAL:
    #PLOT STAR CENTER OF MASS
    if Num_stars != 0:#only calculate if there are stars
        centroid_z = 0
        for j in range(len(grid_counts)):
            centroid_z += z[j]*grid_counts[j]
        centroid_z = centroid_z / Num_stars
        
        # #Find center of distribution / max value and index:
        # i = 0
        # max_bool = False
        # while max_bool == False:
        #     for j in range(len(rho)):
        #         if rho[j] > rho[i]: #if you come across an index j that points to a larger value..
        #             #then set i equal to j
        #             i = j 
        #             #break
        #         else:
        #             max_index = i
        #             max_bool = True

        # max_rho = rho[max_index]

        # #Other method to accumulate left and right sides:
        # for star in stars:
        #     star.x = star.x - z[max_index] #shift
        #     star.reposition(L) #reposition

        # Reposition the center of mass
        # grid_counts = NB.grid_count(stars,L,z)
        # centroid_z = 0
        # for j in range(len(grid_counts)):
        #     centroid_z += z[j]*grid_counts[j]
        # centroid_z = centroid_z / Num_stars

        # for star in stars:
        #     star.x = star.x - centroid_z #shift
        #     star.reposition(L) #reposition

        grid_counts = NB.grid_count(stars,L,z)
        rho_part = (grid_counts/dz)*sigma 
        #Add the density from the FDM
        rho_FDM = mu*np.absolute(chi)**2 
        rho = rho_FDM + rho_part


        centroid_z = 0
        for j in range(len(grid_counts)):
            centroid_z += z[j]*grid_counts[j]
        centroid_z = centroid_z / Num_stars
        ax['lower right'].scatter(centroid_z,0,s = 100,c = "r",marker = "o")


    #ACCELERATIONS
    Part_force = -GF.gradient(GF.fourier_potential(rho_part,L,type = type,G_tilde = G_tilde),L,type = type)
    FDM_force = -GF.gradient(GF.fourier_potential(rho_FDM,L,type=type,G_tilde = G_tilde),L,type = type)
    a1 = np.abs([np.max(Part_force),np.min(Part_force)])
    a2 = np.abs([np.max(FDM_force),np.min(FDM_force)])
    a_max = np.max(np.append(a1,a2))*2
    ax['far right'].plot(z, Part_force, label = "Particle Contribution")
    ax['far right'].plot(z, FDM_force, label = "FDM Contribution")
    ax['far right'].set_ylim(-a_max,a_max)
    ax['far right'].set_title("Force contributions",fontsize = 15)
    ax['far right'].legend(fontsize = 20)

    # THE FDM
    #ax['upper left'].plot(z,chi.real, label = "Re[$\\chi$]")
    #ax['upper left'].plot(z,chi.imag, label = "Im[$\\chi$]")
    #rho_FDM = np.abs(chi)**2 #already calculated this
    phi_FDM = GF.fourier_potential(rho_FDM,L,type = type,G_tilde = G_tilde)
    ax['upper left'].plot(z,phi_FDM,label = "$\\Phi_{FDM}$ [Fourier perturbation]")
    ax['upper left'].plot(z,rho_FDM,label = "$\\rho_{FDM} = \\chi \\chi^*$")
    #ax['upper left'].set_ylim([-y00_max, y00_max] )
    ax['upper left'].set_xlabel("$z = x/L$")
    ax['upper left'].legend(fontsize = 15)
    ax['upper left'].set_title("Non-Dimensional Densities and Potentials",fontsize = 15)

    #PHASE SPACE CALCULATION:
    #Don't calculate if sim_choice1 == '2'
    eta=(z[-1]-z[0])/np.sqrt(np.pi*len(chi)/2) #resolution for Husimi
    k = 2*np.pi*np.fft.fftfreq(len(z),dz)
    #rescale wavenumber k to velocity v:
    hbar = 1
    v = k*(hbar/mu)
    x_min, x_max = np.min(z), np.max(z)
    v_min, v_max = np.min(v), np.max(v)
    F = FDM_OG.Husimi_phase(chi,z,L,eta)
    max_F = np.max(F)/2
    y_max = 2
    ax['upper right'].imshow(F,extent = (x_min,x_max,v_min,v_max),cmap = cm.coolwarm, norm = Normalize(0,max_F), aspect = (x_max-x_min)/(2*y_max))
    ax['upper right'].set_xlim(x_min,x_max)
    ax['upper right'].set_ylim(-y_max,y_max) #[v_min,v_max])
    ax['upper right'].set_xlabel("$z = x/L$")
    ax['upper right'].set_title("Phase Space Distributions", fontsize = 15)
        
    ##############################################3
    # THE PARTICLES
    #rho_part = (grid_counts/dz)*sigma #already calculated this
    phi_part = GF.fourier_potential(rho_part,L,type = type,G_tilde = G_tilde)
    ax['lower left'].plot(z,phi_part,label = "$\\Phi_{Particles}$ [Fourier perturbation]")
    ax['lower left'].plot(z,rho_part,label = "$\\rho_{Particles}$")
    #ax['lower left'].set_xlim(-L/2,L/2)
    #ax['lower left'].set_ylim(-y10_max,y10_max)
    ax['lower left'].legend(fontsize = 15)

    #Plot the Phase Space distribution
    if Num_stars != 0:
        x_s = stars.x
        v_s = stars.v
        xy = np.vstack([x_s,v_s])
        z = gaussian_kde(xy)(xy)
        ax['lower right'].scatter(x_s,v_s,s = 1,c=z,label = "Particles")
        #ax['lower right'].set_ylim(-y11_max,y11_max)
        ax['lower right'].set_xlim(-L/2,L/2)
        ax['lower right'].legend(fontsize = 15)
        
        for i in range(5):
            ax['lower right'].scatter(stars_x[i], stars_v[i], s = 50, marker = 'o')
        
        #ax['lower right'].scatter([stars_x[i] for i in range(5)],[stars_v[i] for i in range(5)], c = 'black', s = 50, marker = 'o')
        
        ax['lower right'].scatter(centroid_z,0,s = 100,c = "r",marker = "o")
    
    plt.show()

    return rho_part, rho_FDM



def fit_func(z,*pars):
        C = pars[0]
        a = pars[1]
        return C/(z*(z+a)**2)

def new_fit_func(z,*pars):
    a0,a1,a2 = pars
    og = a0/(z*(z+a1)**2)
    correction = -a2/z
    return og - correction

def new_new_fit_func(z,*pars):
    a0,a1,a2 = pars
    og = a0/((z**a2) * (z+a1)**2)
    return og

def new_new_new_fit_func(z,*pars):
    a0,a1,a2,a3 = pars
    og = a0/((z**a2) * (z+a1)**a3)
    return og

def curve_fitting(L, z_whole,rho_whole,type = 'Periodic', G_tilde = None):#z_left,z_right,rho_left,rho_right):
    z_whole = z_whole + 0.001
    ###################################################
    # Curve Fitting
    ###################################################

    # Note: Before fitting, we have to shift the z_array up a tiny amount
    # ...b/c if there is a z = 0.0, there will be a divide by zero error

    cols=0
    popt = None
    try:
        guess_params = [1,1]
        popt1,pcov1 = opt.curve_fit(fit_func,z_whole,rho_whole,guess_params,maxfev = 5000)
        print("Check")
        popt = popt1
        cols += 1 
        try: 
            guess_params = [1,1,0]#np.append(popt,0)
            popt2,pcov2 = opt.curve_fit(new_fit_func,z_whole,rho_whole,guess_params)
            print("Check")
            popt = popt2
            cols += 1 
            try: 
                guess_params = [1,1,0]#np.append(popt,0)
                popt3,pcov3 = opt.curve_fit(new_new_fit_func,z_whole,rho_whole,guess_params)
                print("Check")
                popt = popt3
                cols += 1 
                try:
                    guess_params = np.append(popt3,2)# [1,1,0,2]
                    popt4,pcov4 = opt.curve_fit(new_new_new_fit_func,z_whole,rho_whole,guess_params,maxfev = 5000)
                    print("Check")
                    popt = popt4
                    cols += 1 
                except:
                    popt = popt4
                    pass            
            except:
                popt = popt3
                pass
        except:
            popt = popt2
            pass
    except:
        popt = popt1
        pass
    
    # plot = True
    # if cols == 0:
    #     plot = False
    #     #cols = 1
    print(f"#columns = {cols}")
    if cols >= 1:
        fig,ax = plt.subplots(2,cols,figsize = (30,10))
        print(ax)
        #special cases when cols == 1:
        if cols == 1:
            ax0 = ax[0]
            ax1 = ax[1]
        else:
            ax0 = ax[0,0]
            ax1 = ax[1,0]     
        plt.suptitle("Density vs |z| with Curve fit",fontsize = 25)
        
        fit_rho = fit_func(z_whole,*popt1)
        ax0.plot(z_whole,rho_whole)
        ax0.plot(z_whole,fit_rho,'r--',label="Curve Fit")
        ax0.set_xlim(-0.1,1.1)#L/2)
        ax0.text(L/8,max(rho_whole)*3/4, "$f(|z|) = \\frac{a_0}{|z|(|z|+a_1)^2}$",fontsize = 30)
        ax0.text(L/8,max(rho_whole)*1/2, f"$a_0 = {popt1[0]}$",fontsize = 15)
        ax0.text(L/8,0.85*max(rho_whole)*1/2, f"$a_1 = {popt1[1]}$",fontsize = 15)
        
        ax0.legend(fontsize = 25)

        residuals = fit_rho-rho_whole
        resid_y_max = np.max(residuals)
        ax1.plot(z_whole,residuals,"r.--")
        ax1.set_xlim(-0.1,1.1)#L/2)
        ax1.set_ylim((-resid_y_max,resid_y_max))
        #ax[1,0].legend()

        chi2 = 0
        for i in range(len(residuals)):
            chi2 += (residuals[i])**2 / fit_rho[i]
        ax1.text(L/4, 0.8*np.max(residuals), f"$chi^2$ = {chi2}")

    if cols >= 2:
        fit_rho = new_fit_func(z_whole,*popt2)
        ax[0,1].plot(z_whole,rho_whole)
        ax[0,1].plot(z_whole,fit_rho,'r--',label="Curve Fit")
        ax[0,1].set_xlim(-0.1,1.1)#L/2)
        ax[0,1].text(L/8,max(rho_whole)*3/4, "$f(|z|) = \\frac{a_0}{|z|(|z|+a_1)^2}-\\frac{a_2}{|z|}$",fontsize = 30)
        ax[0,1].text(L/8,max(rho_whole)*1/2, f"$a_0 = {popt2[0]}$",fontsize = 15)
        ax[0,1].text(L/8,0.85*max(rho_whole)*1/2, f"$a_1 = {popt2[1]}$",fontsize = 15)
        ax[0,1].text(L/8,0.70*max(rho_whole)*1/2, f"$a_2 = {popt2[2]}$",fontsize = 15)
        
        ax[0,1].legend(fontsize = 25)

        residuals = fit_rho-rho_whole
        ax[1,1].plot(z_whole,residuals,"r.--")
        ax[1,1].set_xlim(-0.1,1.1)
        ax[1,1].set_ylim((-resid_y_max,resid_y_max))
    
        chi2 = 0
        for i in range(len(residuals)):
            chi2 += (residuals[i])**2 / fit_rho[i]
        ax[1,1].text(L/4, 0.8*np.max(residuals), f"$chi^2$ = {chi2}")

    if cols >= 3:
        fit_rho = new_new_fit_func(z_whole,*popt3)
        ax[0,2].plot(z_whole,rho_whole)
        ax[0,2].plot(z_whole,fit_rho,'r--',label="Curve Fit")
        ax[0,2].set_xlim(-0.1,1.1)#L/2)
        ax[0,2].text(L/8,max(rho_whole)*3/4, "$f(|z|) = \\frac{a_0}{|z|^{a_2}(|z|+a_1)^2}$",fontsize = 30)
        ax[0,2].text(L/8,max(rho_whole)*1/2, f"$a_0 = {popt3[0]}$",fontsize = 15)
        ax[0,2].text(L/8,0.85*max(rho_whole)*1/2, f"$a_1 = {popt3[1]}$",fontsize = 15)
        ax[0,2].text(L/8,0.70*max(rho_whole)*1/2, f"$a_2 = {popt3[2]}$",fontsize = 15)
        
        ax[0,2].legend(fontsize = 25)

        residuals = fit_rho-rho_whole
        ax[1,2].plot(z_whole,residuals,"r.--")
        ax[1,2].set_xlim(-0.1,1.1)#L/2)
        ax[1,2].set_ylim((-resid_y_max,resid_y_max))
    
        #ax[1].legend()

        chi2 = 0
        for i in range(len(residuals)):
            chi2 += (residuals[i])**2 / fit_rho[i]
        ax[1,2].text(L/4, 0.8*np.max(residuals), f"$chi^2$ = {chi2}")

    if cols >= 4:    
        fit_rho = new_new_new_fit_func(z_whole,*popt4)
        ax[0,3].plot(z_whole,rho_whole)
        ax[0,3].plot(z_whole,fit_rho,'r--',label="Curve Fit")
        ax[0,3].set_xlim(-0.1,1.1)#L/2)
        ax[0,3].text(L/8,max(rho_whole)*3/4, "$f(|z|) = \\frac{a_0}{|z|^{a_2}(|z|+a_1)^{a_3}}$",fontsize = 30)
        ax[0,3].text(L/8,max(rho_whole)*1/2, f"$a_0 = {popt4[0]}$",fontsize = 15)
        ax[0,3].text(L/8,0.85*max(rho_whole)*1/2, f"$a_1 = {popt4[1]}$",fontsize = 15)
        ax[0,3].text(L/8,0.70*max(rho_whole)*1/2, f"$a_2 = {popt4[2]}$",fontsize = 15)
        ax[0,3].text(L/8,0.55*max(rho_whole)*1/2, f"$a_3 = {popt4[3]}$",fontsize = 15)
        
        ax[0,3].legend(fontsize = 25)

        residuals = fit_rho-rho_whole
        ax[1,3].plot(z_whole,residuals,"r.--")
        ax[1,3].set_xlim(-0.1,1.1)#L/2)
        ax[1,3].set_ylim((-resid_y_max,resid_y_max))
    
        #ax[1].legend()

        chi2 = 0
        for i in range(len(residuals)):
            chi2 += (residuals[i])**2 / fit_rho[i]
        ax[1,3].text(L/4, 0.8*np.max(residuals), f"$chi^2$ = {chi2}")

    plt.show()

    ###################################################
    fit_rho = np.append(fit_rho[::-1],fit_rho)#/ 2 #have to divide by 2 becasue we are double counting on the grid
    fit_z = np.linspace(-L/2,L/2,len(fit_rho))
    if type == 'Isolated':
        z_long = np.linspace(-L,L,2*len(fit_rho)-1)
        G = np.abs(z_long)
        G_tilde = np.fft.fft(G)
        plt.plot(G)
        plt.show()
    fit_phi = GF.fourier_potential(fit_rho,L,type = type,G_tilde = G_tilde) 
    
    fig,ax = plt.subplots(2,1, figsize = (8,10), gridspec_kw = {"height_ratios": [2,1]})
    plt.suptitle("Gravitational Potential in the Box")

    ax[0].plot(fit_z,fit_phi,label = "Analytic Model")
    rho_whole = np.append(rho_whole[::-1],rho_whole)#np.append(rho_left,rho_right)
    z_whole = np.append(-z_whole[::-1],z_whole)
    ax[0].plot(z_whole,GF.fourier_potential(rho_whole,L,type,G_tilde), label = "Exact NBody Potential")
    ax[0].legend()

    ax[1].plot(fit_z,fit_phi-GF.fourier_potential(rho_whole,L),'r,--')
    ax[1].set_title("Residuals")
    plt.show()

    return popt

def pars_track(params,num_s):
    import numpy as np

    p0 = []
    p1 = []
    p2 = []
    p3 = []
    for p in params:
        if len(p) >= 1: 
            p0.append(p[0])
            if len(p)>= 2:
                p1.append(p[1])
                if len(p)>= 3:
                    p2.append(p[2])
                    if len(p)>= 4:
                        p3.append(p[3])
                    else:
                        p3.append(None)
                else:
                    p2.append(None)
            else:
                p1.append(None)
        else:
            p0.append(None)

    array1 = [[n,p] for n,p in zip(num_s,p0)]
    array2 = [[n,p] for n,p in zip(num_s,p1)]
    array3 = [[n,p] for n,p in zip(num_s,p2)]
    array4 = [[n,p] for n,p in zip(num_s,p3)]
    array1.sort()
    array2.sort()
    array3.sort()
    array4.sort()
    array1 = np.array(array1)
    array2 = np.array(array2)
    array3 = np.array(array3)
    array4 = np.array(array4)

    print(array1)
    print(array2)
    print(array3)
    print(array4)

    fig,ax = plt.subplots(1,4, figsize = (15,5))

    ax[0].plot(array1[1:,0],array1[1:,1],"o--",color = "red")
    ax[0].set_title("$a_0$")
    ax[1].plot(array2[1:,0],array2[1:,1],"o--",color = "blue")
    ax[1].set_title("$a_1$")
    ax[2].plot(array3[1:,0],array3[1:,1],"o--",color = "green")
    ax[2].set_title("$a_2$")
    ax[3].plot(array4[1:,0],array4[1:,1],"o--",color = "orange")
    ax[3].set_title("$a_3$")

    plt.show()
