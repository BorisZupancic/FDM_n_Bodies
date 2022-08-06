import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
import os
import subprocess
import cv2 
from PIL import Image 
import scipy.optimize as opt

import OneD.Wave as Wave
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
def analysis(folder: str):#,*args):
    # if len(args) == 0:
    #     #r,m,Num_bosons,sigma,Num_stars = [0,0,0,0,0]
    #     print("")
    #     print("Input r, m, Num_bosons, sigma, Num_stars: ")
    #     r = float(input())
    #     m = float(input())
    #     Num_bosons = int(input())
    #     sigma = float(input())
    #     Num_stars = int(input())
    # else:
    #     r,m,Num_bosons,sigma,Num_stars = args
    #print(os.getcwd())
    #os.chdir(where_folder +"/"+folder)
    #print(os.getcwd())
    os.chdir(folder)
    print(os.getcwd())

    #Load in the necessary quantities:
    Properties = np.loadtxt("Properties.csv", dtype = str, delimiter = ",")
    L,mu,Num_bosons,sigma,Num_stars,r,N = [np.float(Properties[1::,1][i]) for i in range(7)]
    print(r,Num_stars)
    # m = Properties[2,1]
    # Num_bosons = Properties[3,1]
    # sigma = Properties[4,1]
    # Num_stars = Properties[5,1]
    # r = Properties[6,1]

    percent_FDM = Num_bosons*mu / (Num_bosons*mu + Num_stars*sigma)

    #L = 2
    #N = 10**3
    N = int(N)
    z = np.linspace(-L/2,L/2,N)
    dz = z[1]-z[0]

    if Num_bosons == 0:
        #folder = f"{Num_stars}ParticlesOnly_Snapshots"
        stars_x = np.loadtxt("StarsOnly_Pos.csv", dtype = float, delimiter=",")
        stars_v = np.loadtxt("StarsOnly_Vel.csv", dtype = float, delimiter=",")
        K_Energies = np.loadtxt("K_star_Energies.csv", dtype = float,delimiter = ",")
        W_Energies = np.loadtxt("W_star_Energies.csv", dtype = float,delimiter = ",")
        K_5stars_Energies = np.loadtxt("K_5stars_Energies.csv", dtype = float,delimiter = ",")
        W_5stars_Energies = np.loadtxt("W_5stars_Energies.csv", dtype = float,delimiter = ",")
        chi = np.loadtxt(f"Chi.csv", dtype = complex, delimiter=",")
        #chi = np.zeros_like(z)
        centroids = np.loadtxt("Centroids.csv",dtype = float, delimiter=',')
    elif Num_stars == 0:
        #folder = f"OnlyFDM_r{r}_Snapshots"
        chi = np.loadtxt("Chi.csv", dtype = complex, delimiter=",")
        Ks_FDM = np.loadtxt("K_FDM_storage.csv",dtype=float,delimiter=",")
        Ws_FDM = np.loadtxt("W_FDM_storage.csv",dtype=float,delimiter=",")
        centroids = None
        stars_x = None
        star_v = None
        K_5stars_Energies = None
        W_5stars_Energies = None 
        K_Energies = None 
        W_Energies = None
    elif Num_bosons!=0 and Num_stars !=0:
        #folder = f"FDM{percent_FDM}_r{r}_Snapshots"
        stars_x = np.loadtxt("Stars_Pos.csv", dtype = float, delimiter=",")
        stars_v = np.loadtxt("Stars_Vel.csv", dtype = float, delimiter=",")
        chi = np.loadtxt("Chi.csv", dtype = complex, delimiter=",")
        K_Energies = np.loadtxt("K_star_Energies.csv", dtype = float,delimiter = ",")
        W_Energies = np.loadtxt("W_star_Energies.csv", dtype = float,delimiter = ",")
        K_5stars_Energies = np.loadtxt("K_5stars_Energies.csv", dtype = float,delimiter = ",")
        W_5stars_Energies = np.loadtxt("W_5stars_Energies.csv", dtype = float,delimiter = ",")
        
        centroids = np.loadtxt("Centroids.csv",dtype = float, delimiter=',')
        Ks_FDM = np.loadtxt("K_FDM_storage.csv",dtype=float,delimiter=",")
        Ws_FDM = np.loadtxt("W_FDM_storage.csv",dtype=float,delimiter=",")
    

    #Setup our data post-import:
    if stars_x is not None:
        stars = [NB.star(i,sigma,stars_x[i],stars_v[i]) for i in range(len(stars_x))]
    else: 
        stars = []
    m=mu #M_s = 1
    rho_part, rho_FDM = plot_FDMnBodies(z,L,m,mu,sigma,r,stars,chi)
    phi_part = GF.fourier_potentialV2(rho_part,L)
    phi_FDM = GF.fourier_potentialV2(rho_FDM,L)
    
    #######################################################
    ### FDM ANALAYSIS #######################################
    if Num_bosons != 0:
        FDM.plot_Energies(Ks_FDM,Ws_FDM)

        v_dist = FDM.v_distribution(z,L,chi,r,mu)

        FDM_z_rms,FDM_v_rms = FDM.rms_stuff(z,L,chi,v_dist,mu)
        print(f"z_rms = {FDM_z_rms}")
        print(f"v_rms = {FDM_v_rms}")

    #######################################################
    ### NBODY ANALYSIS ######################################
    if Num_stars != 0:
        indices = [0,100,200,400,800,1600,3200,6400]
        NBody.plot_centroids(indices,centroids)
        
        #Calculate rms velocity and position
        z_rms,v_rms = NBody.rms_stuff(sigma,stars,phi_part,L,z,dz)

        NBody.v_distribution(stars,L)

        NBody.select_stars_plots(z,K_5stars_Energies,W_5stars_Energies)

        NBody.all_stars_plots(K_Energies,W_Energies)

        # plot_DeltaE(K_Energies,W_Energies)

        #Hist the final energies:
        Energies = K_Energies+W_Energies
        n_bins = int(np.floor(np.sqrt(len(Energies))))
        plt.hist(Energies[-1,:],bins = n_bins)#100)
        plt.show()

        NBz_left,NBz_right,NBrho_left,NBrho_right = NBody.rho_distribution(z,rho_part)

        popt = curve_fitting(L,z,NBz_left,NBz_right,NBrho_left,NBrho_right)
        print(f"fit params = {popt}")

    if 'FDM_z_rms' in locals() and 'FDM_v_rms' in locals() and 'z_rms' in locals() and 'v_rms' in locals():
        return r, Num_stars, FDM_z_rms, FDM_v_rms, z_rms,v_rms
    elif 'z_rms' in locals() and 'v_rms' in locals():
        return r, Num_stars, z_rms, v_rms
    elif 'FDM_z_rms' in locals() and 'FDM_v_rms' in locals():
        return r, Num_stars, FDM_z_rms, FDM_v_rms

########################################################
# ALL THE ACTUAL FUNCTIONS:
########################################################
def plot_FDMnBodies(z,L,m,mu,sigma,r,stars,chi):
    stars_x = [star.x for star in stars]
    stars_v = [star.v for star in stars]
    Num_stars = len(stars)

    #rescale wavenumber k to velocity v:
    dz = z[1]-z[0]
    k = 2*np.pi*np.fft.fftfreq(len(z),dz)
    hbar = 1
    v = k*(hbar/mu)
    x_min, x_max = np.min(z), np.max(z)
    v_min, v_max = np.min(v), np.max(v)


    #Calculate Particle distribution on Mesh
    if Num_stars !=0:
        stars = [NB.star(i,sigma,stars_x[i],stars_v[i]) for i in range(len(stars_x))]
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

        for star in stars:
            star.x = star.x - centroid_z #shift
            star.reposition(L) #reposition

        grid_counts = NB.grid_count(stars,L,z)
        rho_part = (grid_counts/dz)*sigma 
        #Add the density from the FDM
        rho_FDM = mu*np.absolute(chi)**2 
        rho = rho_FDM + rho_part

        # centroid_z = 0
        # for j in range(len(grid_counts)):
        #     centroid_z += z[j]*grid_counts[j]
        # centroid_z = centroid_z / Num_stars
        ax['lower right'].scatter(centroid_z,0,s = 100,c = "r",marker = "o")


    #ACCELERATIONS
    Part_force = -GF.fourier_gradient(GF.fourier_potentialV2(rho_part,L),L)
    FDM_force = -GF.fourier_gradient(GF.fourier_potentialV2(rho_FDM,L),L)
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
    phi_FDM = GF.fourier_potentialV2(rho_FDM,L)
    ax['upper left'].plot(z,phi_FDM,label = "$\\Phi_{FDM}$ [Fourier perturbation]")
    ax['upper left'].plot(z,rho_FDM,label = "$\\rho_{FDM} = \\chi \\chi^*$")
    #ax['upper left'].set_ylim([-y00_max, y00_max] )
    ax['upper left'].set_xlabel("$z = x/L$")
    ax['upper left'].legend(fontsize = 15)
    ax['upper left'].set_title("Non-Dimensional Densities and Potentials",fontsize = 15)

    #PHASE SPACE CALCULATION:
    #Don't calculate if sim_choice1 == '2'
    eta = 10*r #resolution for Husimi
    k = 2*np.pi*np.fft.fftfreq(len(z),dz)
    #rescale wavenumber k to velocity v:
    hbar = 1
    v = k*(hbar/mu)
    x_min, x_max = np.min(z), np.max(z)
    v_min, v_max = np.min(v), np.max(v)
    F = Wave.Husimi_phase(chi,z,dz,L,eta)
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
    phi_part = GF.fourier_potentialV2(rho_part,L)
    ax['lower left'].plot(z,phi_part,label = "$\\Phi_{Particles}$ [Fourier perturbation]")
    ax['lower left'].plot(z,rho_part,label = "$\\rho_{Particles}$")
    #ax['lower left'].set_xlim(-L/2,L/2)
    #ax['lower left'].set_ylim(-y10_max,y10_max)
    ax['lower left'].legend(fontsize = 15)

    #Plot the Phase Space distribution
    if Num_stars != 0:
        x_s = np.array([star.x for star in stars])
        v_s = np.array([star.v for star in stars])
        ax['lower right'].scatter(x_s,v_s,s = 1,label = "Particles")
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

def curve_fitting(L, z, z_left,z_right,rho_left,rho_right):
    rho_whole = rho_left[::-1] + rho_right
    ###################################################
    # Curve Fitting
    ###################################################

    # Note: Before fitting, we have to shift the z_array up a tiny amount
    # ...b/c if there is a z = 0.0, there will be a divide by zero error

    #Just skip the very first element:
    z_right = z_right[2:]#+1E-10
    rho_whole = rho_whole[2:]

    cols=0
    popt = None
    try:
        guess_params = [1,1]
        popt1,pcov1 = opt.curve_fit(fit_func,z_right,rho_whole,guess_params,maxfev = 5000)
        print("Check")
        popt = popt1
        cols += 1 
        try: 
            guess_params = [1,1,0]#np.append(popt,0)
            popt2,pcov2 = opt.curve_fit(new_fit_func,z_right,rho_whole,guess_params)
            print("Check")
            popt = popt2
            cols += 1 
            try: 
                guess_params = [1,1,0]#np.append(popt,0)
                popt3,pcov3 = opt.curve_fit(new_new_fit_func,z_right,rho_whole,guess_params)
                print("Check")
                popt = popt3
                cols += 1 
                try:
                    guess_params = np.append(popt3,2)# [1,1,0,2]
                    popt4,pcov4 = opt.curve_fit(new_new_new_fit_func,z_right,rho_whole,guess_params,maxfev = 5000)
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
        
        fit_rho = fit_func(z_right,*popt1)
        ax0.plot(z_right,rho_whole)
        ax0.plot(z_right,fit_rho,'r--',label="Curve Fit")
        ax0.set_xlim(-0.1,1.1)#L/2)
        ax0.text(L/8,max(rho_whole)*3/4, "$f(|z|) = \\frac{a_0}{|z|(|z|+a_1)^2}$",fontsize = 30)
        ax0.text(L/8,max(rho_whole)*1/2, f"$a_0 = {popt1[0]}$",fontsize = 15)
        ax0.text(L/8,0.85*max(rho_whole)*1/2, f"$a_1 = {popt1[1]}$",fontsize = 15)
        
        ax0.legend(fontsize = 25)

        residuals = fit_rho-rho_whole
        resid_y_max = np.max(residuals)
        ax1.plot(z_right,residuals,"r.--")
        ax1.set_xlim(-0.1,1.1)#L/2)
        ax1.set_ylim((-resid_y_max,resid_y_max))
        #ax[1,0].legend()

        chi2 = 0
        for i in range(len(residuals)):
            chi2 += (residuals[i])**2 / fit_rho[i]
        ax1.text(L/4, 0.8*np.max(residuals), f"$chi^2$ = {chi2}")

    if cols >= 2:
        fit_rho = new_fit_func(z_right,*popt2)
        ax[0,1].plot(z_right,rho_whole)
        ax[0,1].plot(z_right,fit_rho,'r--',label="Curve Fit")
        ax[0,1].set_xlim(-0.1,1.1)#L/2)
        ax[0,1].text(L/8,max(rho_whole)*3/4, "$f(|z|) = \\frac{a_0}{|z|(|z|+a_1)^2}-\\frac{a_2}{|z|}$",fontsize = 30)
        ax[0,1].text(L/8,max(rho_whole)*1/2, f"$a_0 = {popt2[0]}$",fontsize = 15)
        ax[0,1].text(L/8,0.85*max(rho_whole)*1/2, f"$a_1 = {popt2[1]}$",fontsize = 15)
        ax[0,1].text(L/8,0.70*max(rho_whole)*1/2, f"$a_2 = {popt2[2]}$",fontsize = 15)
        
        ax[0,1].legend(fontsize = 25)

        residuals = fit_rho-rho_whole
        ax[1,1].plot(z_right,residuals,"r.--")
        ax[1,1].set_xlim(-0.1,1.1)
        ax[1,1].set_ylim((-resid_y_max,resid_y_max))
    
        chi2 = 0
        for i in range(len(residuals)):
            chi2 += (residuals[i])**2 / fit_rho[i]
        ax[1,1].text(L/4, 0.8*np.max(residuals), f"$chi^2$ = {chi2}")

    if cols >= 3:
        fit_rho = new_new_fit_func(z_right,*popt3)
        ax[0,2].plot(z_right,rho_whole)
        ax[0,2].plot(z_right,fit_rho,'r--',label="Curve Fit")
        ax[0,2].set_xlim(-0.1,1.1)#L/2)
        ax[0,2].text(L/8,max(rho_whole)*3/4, "$f(|z|) = \\frac{a_0}{|z|^{a_2}(|z|+a_1)^2}$",fontsize = 30)
        ax[0,2].text(L/8,max(rho_whole)*1/2, f"$a_0 = {popt3[0]}$",fontsize = 15)
        ax[0,2].text(L/8,0.85*max(rho_whole)*1/2, f"$a_1 = {popt3[1]}$",fontsize = 15)
        ax[0,2].text(L/8,0.70*max(rho_whole)*1/2, f"$a_2 = {popt3[2]}$",fontsize = 15)
        
        ax[0,2].legend(fontsize = 25)

        residuals = fit_rho-rho_whole
        ax[1,2].plot(z_right,residuals,"r.--")
        ax[1,2].set_xlim(-0.1,1.1)#L/2)
        ax[1,2].set_ylim((-resid_y_max,resid_y_max))
    
        #ax[1].legend()

        chi2 = 0
        for i in range(len(residuals)):
            chi2 += (residuals[i])**2 / fit_rho[i]
        ax[1,2].text(L/4, 0.8*np.max(residuals), f"$chi^2$ = {chi2}")

    if cols >= 4:    
        fit_rho = new_new_new_fit_func(z_right,*popt4)
        ax[0,3].plot(z_right,rho_whole)
        ax[0,3].plot(z_right,fit_rho,'r--',label="Curve Fit")
        ax[0,3].set_xlim(-0.1,1.1)#L/2)
        ax[0,3].text(L/8,max(rho_whole)*3/4, "$f(|z|) = \\frac{a_0}{|z|^{a_2}(|z|+a_1)^{a_3}}$",fontsize = 30)
        ax[0,3].text(L/8,max(rho_whole)*1/2, f"$a_0 = {popt4[0]}$",fontsize = 15)
        ax[0,3].text(L/8,0.85*max(rho_whole)*1/2, f"$a_1 = {popt4[1]}$",fontsize = 15)
        ax[0,3].text(L/8,0.70*max(rho_whole)*1/2, f"$a_2 = {popt4[2]}$",fontsize = 15)
        ax[0,3].text(L/8,0.55*max(rho_whole)*1/2, f"$a_3 = {popt4[3]}$",fontsize = 15)
        
        ax[0,3].legend(fontsize = 25)

        residuals = fit_rho-rho_whole
        ax[1,3].plot(z_right,residuals,"r.--")
        ax[1,3].set_xlim(-0.1,1.1)#L/2)
        ax[1,3].set_ylim((-resid_y_max,resid_y_max))
    
        #ax[1].legend()

        chi2 = 0
        for i in range(len(residuals)):
            chi2 += (residuals[i])**2 / fit_rho[i]
        ax[1,3].text(L/4, 0.8*np.max(residuals), f"$chi^2$ = {chi2}")

    plt.show()

    ###################################################
    fig,ax = plt.subplots(2,1, figsize = (8,10), gridspec_kw = {"height_ratios": [2,1]})
    plt.suptitle("Gravitational Potential in the Box")
    fit_rho = np.append(fit_rho[::-1],fit_rho)/ 2 #have to divide by 2 becasue we are double counting on the grid
    fit_phi = GF.fourier_potentialV2(fit_rho,L) 
    fit_z = np.linspace(-L/2,L/2,len(fit_phi))
    ax[0].plot(fit_z,fit_phi,label = "Analytic Model")
    rho = np.append(rho_left,rho_right)
    ax[0].plot(z,GF.fourier_potentialV2(rho,L), label = "Exact NBody Potential")
    ax[0].legend()

    ax[1].plot(fit_z,fit_phi-GF.fourier_potentialV2(rho,L)[4:],'r,--')
    ax[1].set_title("Residuals")
    plt.show()

    return popt