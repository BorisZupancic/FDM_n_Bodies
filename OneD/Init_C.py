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

import OneD.FDM as FDM
import OneD.Global as GF
import OneD.NBody as NB

def startup():
    '''
    A function that starts-up the simulation by surveying for 
    options/choice input. Includes:
    
    - Whether to run under fixed potential, 
    - Type of boundary conditions,
    - Initial conditions, 
    - Length of simulation (dynamical times)
    
    Returns: inputs to survey. 
    '''

    print("")
    print("Do you want a fixed potetial (phi = 0.5*sigma*(2z/L)**2 - 1)? Choose [y/n]")
    fixed_phi = input()
    print(fixed_phi)
    if fixed_phi == 'Y' or fixed_phi == 'y' or fixed_phi == None:
        fixed_phi = True
    if fixed_phi == 'n':
        fixed_phi = False
    print("")
    print("Isolated [1] or Periodic [2] boundary conditions?")
    bc_choice = int(input())
    print(bc_choice)
    print("")
    print("Do you want the full simulation [1] or snapshots [2]? Choose [1/2]")
    sim_choice2 = int(input())
    print(sim_choice2)
    print("")

    print("How long to run for? Enter Integer for number of dynamical times:")
    dynamical_times = int(input())
    print(f"Will run for {dynamical_times} dynamical times")
    print("")


    #Create Initial Conditions:
    print("Initial Conditions: Gaussian, Sine^2, or Spitzer? Enter [1,2,or 3]:")
    ICs = float(input())
    print(ICs)
    print("")

    return fixed_phi, bc_choice, sim_choice2, dynamical_times, ICs

def init(hbar,L_scale,v_scale, ICs):
    '''
    A function to instantiate the FDM and/or particle system. 
    '''

    M_scale = L_scale*v_scale**2

    print("Choose a (non-dimensional) box length:")
    L = float(input())
    print(f"L={L}")
    print("")

    print("Choose percentage (as a decimal) of FDM (by mass)")
    percent_FDM = float(input())
    percent_Particles = 1 - percent_FDM
    print(f"Fraction of Particles (by mass) = {percent_Particles}")
    
    print("")
    if percent_FDM != 1:
        print("How many particles?")
        Num_stars = int(input())
    else:
        Num_stars = 0

    #Set up Grid
    L = L*L_scale #new length. Length of the box
    N = 1*10**3 #default number of grid points
    z = np.linspace(-L/2,L/2,N)
    dz = z[1]-z[0]

    if percent_FDM != 0:
        print("Choose a FDM fuzziness.")
        print("Input a de Broglie wavelength ratio to Characteristic size:")
        lambda_ratio = float(input()) #/R
        print(lambda_ratio)
        print("")
        
    if ICs == 1:
        sigma=0
        if Num_stars!=0:
            sigma = 1*percent_Particles/Num_stars
            print("Choose the standard deviation of the Particle system (as fraction of the box width):")
            Stars_std = float(input())
        else:
            Stars_std = 0.1 #value doesn't matter (just not 0)
        Stars_std = Stars_std*L 

        if percent_FDM != 0:
            print("Choose the standard deviation of the initial FDM distribution (as a fraction of the box width):")
            FDM_std = float(input())
            FDM_std=FDM_std*L
        else:
            FDM_std = None
        stars, chi, r, T_Dynamical, zmax, vmax = gaussian(z, L, lambda_ratio, percent_FDM, FDM_std, sigma, Num_stars, Stars_std)
        mu = 0.5/r
        print("Gaussian ICs instantiated.")
    # elif ICs == 2:
    #     stars,chi = sine2(z, L, Num_bosons, sigma, Num_stars, v_s, L_s)
    #     print("Sine^2 ICs instantiated.")
    elif ICs == 3:
        E0,v_sigma,f0 = .7, .5, .1 #.15, .3, .05
        stars, chi, z_rms, v_rms, z, zmax, vmax, dtau  = Spitzer(Num_stars,percent_FDM,z,E0,v_sigma,f0, lambda_ratio)
        T_Dynamical = z_rms/v_rms
        print("Spitzer ICs instantiated.")

        print("")
        if percent_FDM != 0:
            lambda_deB = zmax / lambda_ratio 
            
            r = lambda_deB*v_rms / (4*np.pi)
            print(f"lambda_deB = {lambda_deB}")
            print(f"Fuzziness: r = {r}")
            mu = 1/(2*r)
            print(f"Mass mu = {mu}")
            print(f"Number of Grid points: {len(z)}")
            
        elif percent_FDM == 0:
            r = 0.5
            mu = 1 #set as default
            R = None
            lambda_deB = None 

    Total_mass = stars.mass*Num_stars + mu*np.sum(np.abs(chi)**2)*dz
    Num_bosons = Total_mass*percent_FDM/mu
    
    print(f"Num_stars = {Num_stars}")
    
    print(f"Num_Bosons = {Num_bosons}")
    print(f"mu = {mu}")
    
    return z, stars, chi, mu, Num_bosons, r, T_Dynamical, zmax, vmax, dtau

########################################################################################

def normal_dist(x,b,std):
    return np.exp(-(x-b)**2/(2*std**2))/(np.sqrt(2*np.pi)*std)

def gaussian(z, L, lambda_ratio, percent_FDM, FDM_std, sigma, Num_stars, Stars_std):
    ########################################################
    # INITIAL SETUP
    ########################################################
    

    print("Total mass of system set to 1")
    Total_mass = 1
    
    

    ####################################################
    # FOR THE PARTICLES
    #Set initial distribution on grid
    b = 0 #center at zero
    
    z_0 = np.random.normal(b,Stars_std,Num_stars) #initial positions sampled from normal distribution
    stars = NB.stars(sigma,z_0,np.zeros_like(z_0))
    zmax = np.max([2*Stars_std, 2*FDM_std])

    stars.reposition(L)

    # Reposition the center of mass
    grid_counts = NB.grid_count(stars,L,z)
    if Num_stars != 0: 
        centroid_z = 0
        for j in range(len(grid_counts)):
            centroid_z += z[j]*grid_counts[j]
        centroid_z = centroid_z / Num_stars

        stars.x = stars.x - centroid_z
        stars.reposition(L)
    
    #Calculate distribution on Mesh
    if Num_stars !=0:
        rho_part = NB.particle_density(stars, L, z, variable_mass = False)
    else:
        rho_part = np.zeros_like(z)

    # FOR THE FDM 
    #Set an initial wavefunction
    b=0
    if percent_FDM != 0:
        FDM_mass = Total_mass*percent_FDM
        #want: dz*np.sum(np.abs(chi)**2 * mu) = FDM_mass
        rho_FDM = normal_dist(z,b,FDM_std)*FDM_mass
        
        #chi = np.sqrt(rho_FDM/mu)
        # psi = np.sqrt(normal_dist(z,b,FDM_std)*Num_bosons)#*Num_particles / (L**3))
        # chi = psi**(3/2) #*L_s; L_s = 1
        
    else:
        chi = np.zeros_like(z)
    #Calculate Collapse Time:
    #rho_FDM = mu*np.absolute(chi)**2 #just norm-squared of wavefunction
    
    

    rho = rho_FDM+rho_part
    rho_avg = np.mean(rho)
    print(rho_avg)
    tau_collapse = 1/(rho_avg)**0.5 / np.sqrt(2) /4
    print(f"(Non-dim) Collapse time: {tau_collapse}")
    t_dynamical = tau_collapse

    vmax = zmax/t_dynamical

    print(f"FDM_std={FDM_std}")
    FDM_zmax = 2*FDM_std
    lambda_deB = FDM_zmax / lambda_ratio 
    
    v_rms = FDM_std / t_dynamical
    #v_rms = 0.1*FDM_zmax / t_dynamical

    r = lambda_deB*v_rms / (4*np.pi)
    mu = 1/(2*r)

    chi = np.sqrt(rho_FDM/mu)
    return stars, chi, r, t_dynamical, zmax, vmax

def sine2(z, L, Num_bosons, sigma, Num_stars, v_s, L_s):
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
#Spitzer ICs, based off Larry's:
def Spitzer(Num_stars, percent_FDM, z, E0, sigma, f0, lambda_ratio):
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

    #z_original = np.copy(z)
    dz = z[1]-z[0]
    #N = len(z)
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
    print(f"Mass from distribution function (via RK4): M = {M}")
     
    L = 2*(z[-1]-z[0])
    z_new = np.append(-z[::-1],z)
    rho_new = np.append(rho[::-1],rho)
    phi_new = np.append(phi[::-1],phi)
    
    z_long = np.linspace(-L,L,2*len(z_new)-1)
    G = 0.5*np.abs(z_long)
    G_tilde = np.fft.fft(G)

    my_phi = GF.fourier_potential(rho_new,L,type='Isolated', G_tilde = G_tilde)
    my_phi = my_phi - np.min(my_phi)

    rhomax = rho[0]
    zmax = z[imax-1]
    from scipy.interpolate import interp1d
    rhointerp = interp1d(z_new,rho_new)
    phiinterp = interp1d(z_new,phi_new)
    print ('cut-off in z', zmax)

    z_rms = np.sqrt(dz*np.sum(rho_new*z_new**2))
    print(f"z_rms = {z_rms}")

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
        stars = NB.stars(m,xIC,vIC) #[NB.star(0,m,x,v) for x,v in zip(xIC,vIC)]
        
        #re-center position and velocity centroids:
        #print("Re-centering position and velocity centroids.")
        z_centroid = np.mean(stars.x)
        v_centroid = np.mean(stars.v)
        #print(f"z_centroid = {z_centroid}, v_centroid = {v_centroid}")
        stars.x += -z_centroid
        stars.v += -v_centroid
        z_centroid = np.mean(stars.x)
        v_centroid = np.mean(stars.v)
        #print(f"z_centroid = {z_centroid}, v_centroid = {v_centroid}")
        
        v_rms = np.std(stars.v)
        z_rms = np.std(stars.x)
        print(f"z_rms = {z_rms}")
        print(f"v_rms = {v_rms}")
        t_dynamical = z_rms/v_rms
        print(f"t_dynamical = {t_dynamical}")

        star_v = v_rms
    else:
        stars = NB.stars(0,[],[])
        star_v = 0

#STEP 3: Create FDM wavefunction (if applicable)
    if percent_FDM!=0:
        print("------Sampling FDM------")
        def DF(phi, v, E0, sigma, f0):
            '''
            Function that will return N by N array for Spitzer DF:
            '''
            phi_kn, v_kn  = np.meshgrid(phi, v, indexing = 'ij')

            E_kn = 0.5*v_kn**2 + phi_kn
            B_kn = np.exp(-E_kn/sigma**2)
            A = np.exp(E0/sigma**2)
            
            f = np.zeros_like(E_kn)
            f[E_kn <= E0] = f0*(A*B_kn[E_kn <= E0] - 1)

            return f, phi_kn, v_kn

        #N_new = np.copy(N)
        z = z_new
        rho = rho_new
        phi = phi_new
        
        #1. Set Default Grid:
        N = 1000 #default
        #print(f"Num Grid Points = {N}")
        z = np.linspace(z[0],z[-1],N)
        L = z[-1]-z[0]
        dz = z[1]-z[0]
        
        alpha = 1.2
        L_new = 2*zmax*alpha
        N_new = int(np.ceil(L_new / dz))
        
        rho = rhointerp(z)
        phi = phiinterp(z)

        #2. Generate phase space distribution:
        beta = 1.2
        vmax = np.sqrt(2.*E0)
        v = np.linspace(-vmax*beta,vmax*beta,N)
        dv = v[1]-v[0]
        
        f, phi_kn, v_kn = DF(phi, v, E0,sigma,f0)
        #calculate v_rms and z_rms
        v_dist = dz*np.sum(f, axis = 1)
        z_dist = dv*np.sum(f, axis = 0)
        v_rms = np.sqrt(dv*np.sum(v_dist*v**2))
        z_rms = np.sqrt(dz*np.sum(z_dist*z**2))
        t_dynamical = z_rms/v_rms
        print(f"z_rms = {z_rms}")
        print(f"v_rms = {v_rms}")
        print(f"t_dynamical = {t_dynamical}")

        
        #3. Re-set grid to proper resolution:
        
        lambda_deB = zmax / lambda_ratio
        r = lambda_deB*v_rms / (4*np.pi)
        mu = 1/(2*r)
        
        N = beta*zmax*vmax / np.pi / r #alpha*beta*zmax*vmax / np.pi / r
        N = int(np.ceil(N))
        
        z = np.linspace(z[0],z[-1],N)
        L = z[-1]-z[0]
        dz = z[1]-z[0]
        
        alpha = 1.2
        L_new = 2*zmax*alpha
        N_new = int(np.ceil(L_new / dz)) + 1
        print(f"Num Grid Points = {N_new}")

        rho = rhointerp(z)
        phi = phiinterp(z) 

        vmax = np.sqrt(2.*E0)
        pmax = np.pi*(N-1)/L_new 

        p = np.linspace(-pmax,pmax,N)
        dp = p[1]-p[0]
        v = p/mu

        f, phi_kn, v_kn = DF(phi, v, E0,sigma,f0)
        
        #4. Make wavefunction
        z_kn,p_kn = np.meshgrid(z, p, indexing = 'ij')
        thetas = np.random.uniform(0,2*np.pi,len(p))
        R_s = np.exp(1j*thetas)
        chi_kn = np.multiply(R_s,np.sqrt(f)*np.exp(1j*z_kn*p_kn))
        chi = np.sum(chi_kn, axis = 1)
        
        #4b) corrections to momentum/velocity profile:
        k_mean = np.sum(np.conj(chi)*(-1j)*np.gradient(chi,dz)) / np.sum(np.abs(chi)**2)
        T_drift = dz*mu/np.abs(k_mean)
        while t_dynamical/T_drift >= 10**(-3):
            chi = chi*np.exp(-1j*k_mean*z)
            k_mean = np.sum(np.conj(chi)*(-1j)*np.gradient(chi,dz)) / np.sum(np.abs(chi)**2)
            T_drift = mu*dz/np.abs(k_mean)
            
        #Extend wavefunction to entire box:
        N_add = int(N_new-len(chi))
        chi = np.append(np.zeros(N_add//2),chi)
        chi = np.append(chi,np.zeros(N_add//2))
        z = np.linspace(-L_new/2,L_new/2,len(chi))
        dz = z[1]-z[0]
        #Re-normalize:
        Normalization = np.sqrt( M / (dz*np.sum(np.abs(chi)**2)) / mu) #np.sqrt( dz*np.sum(rho) / (dz*np.sum(np.abs(chi)**2)) / mu) 
        chi = chi * Normalization

        FDM_v = v_rms

    else:
        chi = np.zeros(N)
        FDM_v = 0

    if percent_FDM!=0 and Num_stars!=0:
        chi = chi / np.sqrt(2)
        stars.mass = stars.mass / 2

    dtau = 0.5*dz/np.sqrt(2*E0)
    #print(dtau)
    #dtau = 0.5*dz/np.max([star_v, FDM_v])
    #print(dtau)
    return stars, chi, z_rms, v_rms, z, zmax, vmax, dtau  #,t_dynamical
