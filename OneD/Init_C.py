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
        print("Variable Star Masses [y/n]? (Split in two)")
        variable_mass = input()
        print("["+variable_mass+"]")
        if variable_mass == 'n':
            print("How to instantiate particle masses? Identical [1] or Proportional to f(E) [2]")
            stars_type = int(input())

            variable_mass = [False]

        elif variable_mass == 'y':
            print("How to instantiate (heavy) particle masses? Identical [1] or Proportional to f(E) [2]")
            stars_type = int(input())
            if stars_type == 2:
                print("Input number of heavy/quasi-particles:")
                num_heavy = int(input())
                variable_mass = [True,num_heavy]
            else:
                
                print("Input fraction of stars to convert to heavy:")
                fraction = input()
                print(fraction)
                # print("Splitting Stars in Two...")
                num, denom = fraction.split('/')
                fraction = float(num)/float(denom)
                num_to_change = int(np.floor(fraction*Num_stars))
                print(f"Number of heavier particles: {num_to_change}")
                variable_mass = [True,fraction,num_to_change]
                
    else:
        Num_stars = 0
        variable_mass=[False]
        stars_type = None

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
    elif percent_FDM == 0:
        lambda_ratio = None
         
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
        dtau = dz/vmax
        print("Gaussian ICs instantiated.")
    # elif ICs == 2:
    #     stars,chi = sine2(z, L, Num_bosons, sigma, Num_stars, v_s, L_s)
    #     print("Sine^2 ICs instantiated.")
    elif ICs == 3:
        E0,v_sigma,f0 = .7, .5, .1 #.15, .3, .05
        choose_mesh=input("Bypass mesh-size [y/n]?")
        if choose_mesh=='y':
            choose_mesh=True
        else:
            choose_mesh=False
            
        stars, chi, z_rms, v_rms, z, zmax, vmax, dtau, variable_mass  = Spitzer(Num_stars,percent_FDM,z,E0,v_sigma,f0, lambda_ratio, variable_mass, stars_type, choose_mesh)
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

    if variable_mass[0] == False:
        Num_stars = len(stars.x)
        mass_part = np.sum(stars.mass)
    elif variable_mass[0] == True:
        Num_stars = len(stars[0].x)+len(stars[1].x)
        mass_part = np.sum(stars[0].mass) + np.sum(stars[1].mass)
    mass_FDM = mu*dz*np.sum(np.abs(chi)**2)
    Total_mass = mass_part + mass_FDM
    Num_bosons = Total_mass*percent_FDM/mu
    
    print(f"Num_stars = {Num_stars}")
    
    print(f"Num_Bosons = {Num_bosons}")
    # print(f"mu = {mu}")
    
    return z, stars, chi, mu, Num_bosons, r, T_Dynamical, zmax, vmax, dtau, variable_mass, stars_type

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
    masses = sigma*np.ones(Num_stars)
    stars = NB.stars(masses,z_0,np.zeros_like(z_0))
    zmax = np.max([Stars_std, FDM_std])

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
def Spitzer(Num_stars, percent_FDM, z, E0, sigma, f0, lambda_ratio, variable_mass, stars_type, choose_mesh = False):
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

    dz = z[1]-z[0]
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
    rhointerp = interp1d(z_new,rho_new)
    phiinterp = interp1d(z_new,phi_new)
    print ('cut-off in z', zmax)

    z_rms = np.sqrt(np.sum(rho_new*z_new**2) / np.sum(rho_new)) 
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

        vmax = np.sqrt(2.*E0) #re-write the absolute max velocity
        
        #re-center position and velocity centroids:
        #print("Re-centering position and velocity centroids.")
        z_centroid = np.mean(xIC)
        v_centroid = np.mean(vIC)
        #print(f"z_centroid = {z_centroid}, v_centroid = {v_centroid}")
        xIC += -z_centroid
        vIC += -v_centroid
        z_centroid = np.mean(xIC)
        v_centroid = np.mean(vIC)
    
        v_rms = np.std(vIC)
        z_rms = np.std(xIC)
        print(f"z_rms = {z_rms}")
        print(f"v_rms = {v_rms}")
        t_dynamical = z_rms/v_rms
        print(f"t_dynamical = {t_dynamical}")

        star_v = v_rms

        #Set-up appropriate grid:    
        alpha = 1.5
        L_new = 2*zmax*alpha
        #dz = 2*(2*z_rms)/(Num_stars**(1/3)) #1000 #int(np.ceil(np.sqrt(Num_stars)))
        # dz = 2*(2*z_rms)/(100000**(1/3))
        # print(f"dz={dz}")
        
        if choose_mesh == True:
            dz = float(input("Input a mesh-size dz:"))
        else:
            # x1_kn,x2_kn = np.meshgrid(stars.x,stars.x)
            # x_diff = x1_kn-x2_kn
            # ave_spacing = np.mean(x_diff,axis=(0,1))
            x_diff = np.ndarray(Num_stars)
            for i in range(Num_stars):
                x = np.delete(xIC,i)
                x_diff[i] = np.mean(np.abs(x-xIC[i]))
            ave_spacing = np.mean(x_diff)
            dz = .025*ave_spacing
            #dz = np.min(x_diff)
        

        N_star = L_new/dz + 1
        N_star = int(N_star)
        print(f"N={N_star}")
        z = np.linspace(-L_new/2,L_new/2,N_star)
        dz = z[1]-z[0]
        print(f"dz={dz}")
        
        #(Re-)Assigning masses:
        if variable_mass[0] == False:
            if stars_type == 1:
                m = M/Num_stars
                masses = m*np.ones(Num_stars)
                stars = NB.stars(masses,xIC,vIC)

            elif stars_type == 2:
                #uniformly sample xIC and vIC
                #then assign masses proportional to f(x,v)
                
                xIC = np.ndarray(Num_stars)
                vIC = np.ndarray(Num_stars)
                Es = np.ndarray(Num_stars)
                i = 0
                while i < Num_stars:
                    xtmp = np.random.uniform(-zmax,zmax)
                    vtmp = np.random.uniform(-vmax,vmax)
                    p = phiinterp(np.abs(xtmp))
                    e = vtmp**2 /2. + p
                    if e<E0:
                        xIC[i] = xtmp 
                        vIC[i] = vtmp
                        Es[i] = e
                        i+=1

                z_centroid = np.mean(xIC)
                v_centroid = np.mean(vIC)
                xIC += -z_centroid
                vIC += -v_centroid
                z_centroid = np.mean(xIC)
                v_centroid = np.mean(vIC)
                
                masses = np.zeros_like(xIC)
                masses[Es<E0] = f0*(np.exp((E0-Es[Es<E0])/sigma**2))
                #re-normalize mass:
                masses = masses*M/np.sum(masses)
                print(np.sum(masses))

                net_momentum = np.sum(masses*vIC)
                epsilon = net_momentum / np.sum(masses)
                vIC = vIC - epsilon 

                net_position = np.sum(masses*xIC)
                epsilon = net_position / np.sum(masses)
                xIC = xIC - epsilon 

                stars = NB.stars(masses,xIC,vIC)

        elif variable_mass[0] == True:
            if stars_type == 1:
                fraction = variable_mass[1]
                num_to_change = variable_mass[2]
                
                Total_mass = M
                sigma1 = (Total_mass/2)/num_to_change #sigma*2
                sigma2 = (Total_mass/2)/(Num_stars-num_to_change) #sigma*fraction

                from numpy.random import default_rng
                rng = default_rng()
                i_s = rng.choice(Num_stars, size=num_to_change, replace=False)
                print(len(i_s))
                part1 = np.array([[stars.x[i],stars.v[i]] for i in i_s])
                part2 = np.array([[stars.x[i],stars.v[i]] for i in range(Num_stars) if i not in i_s])
                masses1 = sigma1*np.ones(len(part1[:,0]))
                masses2 = sigma2*np.ones(len(part2[:,0]))
                stars1 = NB.stars(masses1,part1[:,0],part1[:,1])
                stars2 = NB.stars(masses2,part2[:,0],part2[:,1])

                #re-center position and velocity centroids:
                z1_centroid, z2_centroid = [np.mean(stars1.x),np.mean(stars2.x)]
                v1_centroid, v2_centroid = [np.mean(stars1.v),np.mean(stars2.v)]
                stars1.x += -z1_centroid
                stars1.v += -v1_centroid
                stars2.x += -z2_centroid
                stars2.v += -v2_centroid
                
                stars = [stars1, stars2]
                print(f"len(stars1.x) = {len(stars[0].x)}")
                print(f"len(stars2.x) = {len(stars[1].x)}")
                print(f"stars1.mass = {stars[0].mass}")
                print(f"sigma1 = {sigma1}")
                
                variable_mass = [True,fraction,sigma1,sigma2]
            
            elif stars_type == 2:
                light_mass = (M/2) / Num_stars
                masses = light_mass*np.ones_like(xIC)
                stars2 = NB.stars(masses,xIC,vIC)
                
                #uniformly sample xIC and vIC
                #then assign masses proportional to f(x,v)
                Num_heavy = variable_mass[1]
                xIC = np.ndarray(Num_heavy)
                vIC = np.ndarray(Num_heavy)
                Es = np.ndarray(Num_heavy)
                i = 0
                while i < Num_heavy:
                    xtmp = np.random.uniform(-zmax,zmax)
                    vtmp = np.random.uniform(-vmax,vmax)
                    p = phiinterp(np.abs(xtmp))
                    e = vtmp**2 /2. + p
                    if e<E0:
                        xIC[i] = xtmp 
                        vIC[i] = vtmp
                        Es[i] = e
                        i+=1

                z_centroid = np.mean(xIC)
                v_centroid = np.mean(vIC)
                xIC += -z_centroid
                vIC += -v_centroid
                z_centroid = np.mean(xIC)
                v_centroid = np.mean(vIC)
                
                masses = np.zeros_like(xIC)
                masses[Es<E0] = f0*(np.exp((E0-Es[Es<E0])/sigma**2))
                #re-normalize mass:
                masses = masses*(M/2)/np.sum(masses)
                stars1 = NB.stars(masses,xIC,vIC)

                #correcting BOTH sets of stars:
                net_position = np.sum(stars1.mass*stars1.x) + np.sum(stars2.mass*stars2.x)
                epsilon = net_position / M # np.sum(masses)
                stars1.x = stars1.x - epsilon
                stars2.x = stars2.x - epsilon 

                net_momentum = np.sum(stars1.mass*stars1.v) + np.sum(stars2.mass*stars2.v)
                epsilon = net_momentum / M # np.sum(masses)
                stars1.v = stars1.v - epsilon 
                stars2.v = stars2.v - epsilon 

                net_position = np.sum(stars1.mass*stars1.x) + np.sum(stars2.mass*stars2.x)
                net_momentum = np.sum(stars1.mass*stars1.v) + np.sum(stars2.mass*stars2.v)
                
                stars = [stars1,stars2]

    else:
        stars = NB.stars([],[],[])
        star_v = 0

        N_star=None
    
    
        
            
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
        z = np.linspace(z[0],z[-1],N)
        L = z[-1]-z[0]
        dz = z[1]-z[0]
        
        alpha = 1.5
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
        #calculate v_rms
        v_dist = dz*np.sum(f, axis = 0)
        v_rms = np.sqrt(np.sum(v_dist*v**2) / np.sum(v_dist)) 
        
        
        #3. Re-set grid to proper resolution:
        lambda_deB = zmax / lambda_ratio
        r = lambda_deB*v_rms / (4*np.pi)
        mu = 1/(2*r)
        
        N = beta*zmax*vmax / np.pi / r #alpha*beta*zmax*vmax / np.pi / r
        N = int(np.ceil(N))
        
        if N_star is not None:
            N = np.max([N,N_star])

        z = np.linspace(z[0],z[-1],N)
        L = z[-1]-z[0]
        dz = z[1]-z[0]
        
        alpha = 1.5
        L_new = 2*zmax*alpha
        N_new = int(np.ceil(L_new / dz)) + 1
        print(f"Num Grid Points = {N_new}")

        rho = rhointerp(z)
        phi = phiinterp(z) 

        vmax = np.sqrt(2.*E0)
        pmax = np.pi*(N_new-1)/L_new 

        p = np.linspace(-pmax,pmax,N_new)
        dp = p[1]-p[0]
        v = p/mu

        f, phi_kn, v_kn = DF(phi, v, E0,sigma,f0)
        #calculate v_rms and z_rms
        v_dist = dz*np.sum(f, axis = 0)
        z_dist = dv*np.sum(f, axis = 1)
        v_rms = np.sqrt(np.sum(v_dist*v**2) / np.sum(v_dist)) 
        z_rms = np.sqrt(np.sum(z_dist*z**2) / np.sum(z_dist))
        t_dynamical = z_rms/v_rms
        print(f"z_rms = {z_rms}")
        print(f"v_rms = {v_rms}")
        print(f"t_dynamical = {t_dynamical}")

        # plt.imshow(f.transpose())
        # plt.show()

        #4. Make wavefunction
        z_kn,p_kn = np.meshgrid(z, p, indexing = 'ij')
        thetas = np.random.uniform(0,2*np.pi,len(p))
        R_s = np.exp(1j*thetas)
        chi_kn = np.multiply(R_s,np.sqrt(f)*np.exp(1j*z_kn*p_kn))
        chi = np.sum(chi_kn, axis = 1)
        
        #4b) corrections to momentum/velocity profile:
        chi_tilde = np.fft.fft(chi)
        k = 2*np.pi*np.fft.fftfreq(len(chi),dz)
        k_mean = np.sum(k*np.abs(chi_tilde)**2) / np.sum(np.abs(chi_tilde)**2)
        print(f"k_mean = {k_mean}")

        chi = chi*np.exp(-1j*k_mean*z)
        
        chi_tilde = np.fft.fft(chi)
        k = 2*np.pi*np.fft.fftfreq(len(chi),dz)
        k_mean = np.sum(k*np.abs(chi_tilde)**2) / np.sum(np.abs(chi_tilde)**2)
        print(f"k_mean = {k_mean}")
                
        #Extend wavefunction to entire box:
        N_add = int(N_new-len(chi))
        chi = np.append(np.zeros(N_add//2),chi)
        chi = np.append(chi,np.zeros(N_add//2))
        z = np.linspace(-L_new/2,L_new/2,len(chi))
        dz = z[1]-z[0]
        #Re-normalize:
        Normalization = np.sqrt( M / (dz*np.sum(np.abs(chi)**2)) / mu) #np.sqrt( dz*np.sum(rho) / (dz*np.sum(np.abs(chi)**2)) / mu) 
        chi = chi * Normalization

    else:
        chi = np.zeros(len(z))
        
    if percent_FDM!=0 and Num_stars!=0:
        chi = chi / np.sqrt(2)
        stars.mass = stars.mass / 2

    dtau = 0.5*dz/np.sqrt(2*E0)
    
    return stars, chi, z_rms, v_rms, z, zmax, vmax, dtau, variable_mass  #,t_dynamical
