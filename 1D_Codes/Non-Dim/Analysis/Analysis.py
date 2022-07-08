import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
import os
import subprocess
import cv2 
from PIL import Image 
import scipy.optimize as opt

#Import My Library
My_Package_PATH = "/home/boris/Documents/Research/FDM_n_Bodies"
import sys
sys.path.insert(1, My_Package_PATH)
import OneD.WaveNonDim as ND
import OneD.NBody as NB
import OneD.GlobalFuncs as GF

#Set up Directory for saving files/images/videos
# Will not rename this again
dirExtension = "1D_Codes/Non-Dim/Analysis"
Directory = os.getcwd()#+"/"+dirExtension #os.curdir() #"/home/boris/Documents/Research/Coding/1D codes/Non-Dim"
print(Directory)

def analysis(*args):
    if len(args) == 0:
        #r,m,Num_bosons,sigma,Num_stars = [0,0,0,0,0]
        print("")
        print("Input r, m, Num_bosons, sigma, Num_stars: ")
        r = float(input())
        m = float(input())
        Num_bosons = int(input())
        sigma = float(input())
        Num_stars = int(input())
    else:
        r,m,Num_bosons,sigma,Num_stars = args

    mu = m #M_scale = 1
    percent_FDM = Num_bosons*mu / (Num_bosons*mu + Num_stars*sigma)

    L = 2
    N = 10**3
    z = np.linspace(-L/2,L/2,N)
    dz = z[1]-z[0]

    if Num_bosons == 0:
        folder = f"{Num_stars}ParticlesOnly_Snapshots"
        stars_x = np.loadtxt(folder+"/"+f"StarsOnly_Pos.csv", dtype = float, delimiter=",")
        stars_v = np.loadtxt(folder+"/"+f"StarsOnly_Vel.csv", dtype = float, delimiter=",")
        K_Energies = np.loadtxt(folder+"/"+"K_Energies.csv", dtype = float,delimiter = ",")
        W_Energies = np.loadtxt(folder+"/"+"W_Energies.csv", dtype = float,delimiter = ",")
        K_5stars_Energies = np.loadtxt(folder+"/"+"K_5stars_Energies.csv", dtype = float,delimiter = ",")
        W_5stars_Energies = np.loadtxt(folder+"/"+"W_5stars_Energies.csv", dtype = float,delimiter = ",")
    
        chi = np.loadtxt(folder+"/"+f"Chi.csv", dtype = complex, delimiter=",")
        #chi = np.zeros_like(z)
        centroids = np.loadtxt(folder+"/"+"Centroids.csv",dtype = float, delimiter=',')
    elif Num_stars == 0:
        folder = f"OnlyFDM_r{r}_Snapshots"
        chi = np.loadtxt(folder+"/"+f"FuzzyOnlyChi_r{r}.csv", dtype = complex, delimiter=",")
        Energies = None
    elif Num_bosons!=0 and Num_stars !=0:
        folder = f"FDM{percent_FDM}_r{r}_Snapshots"
        stars_x = np.loadtxt(folder+"/"+f"Stars_Pos.csv", dtype = float, delimiter=",")
        stars_v = np.loadtxt(folder+"/"+f"Stars_Vel.csv", dtype = float, delimiter=",")
        chi = np.loadtxt(folder+"/"+f"Chi.csv", dtype = complex, delimiter=",")
        K_Energies = np.loadtxt(folder+"/"+"K_Energies.csv", dtype = float,delimiter = ",")
        W_Energies = np.loadtxt(folder+"/"+"W_Energies.csv", dtype = float,delimiter = ",")
        centroids = np.loadtxt(folder+"/"+f"Centroids.csv",dtype = float, delimiter=',')

    # stars_x = np.loadtxt(f"Stars_Pos.csv", dtype = float, delimiter=",")
    # stars_v = np.loadtxt(f"Stars_Vel.csv", dtype = float, delimiter=",")
    # chi = np.loadtxt(f"Chi.csv", dtype = complex, delimiter=",")

    plt.figure()
    plt.title("Centroid over time")
    indices = [0,100,200,400,800,1600,3200,6400]
    plt.plot(indices,centroids,'bo-')
    plt.show()



    #rescale wavenumber k to velocity v:
    k = 2*np.pi*np.fft.fftfreq(len(z),dz)
    hbar = 1
    v = k*(hbar/m)
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
        # centroid_z = 0
        # for j in range(len(grid_counts)):
        #     centroid_z += z[j]*grid_counts[j]
        # centroid_z = centroid_z / Num_stars
        
        #Find center of distribution / max value and index:
        i = 0
        max_bool = False
        while max_bool == False:
            for j in range(len(rho)):
                if rho[j] > rho[i]: #if you come across an index j that points to a larger value..
                    #then set i equal to j
                    i = j 
                    #break
                else:
                    max_index = i
                    max_bool = True

        max_rho = rho[max_index]

        #Other method to accumulate left and right sides:
        for star in stars:
            star.x = star.x - z[max_index] #shift
            star.reposition(L) #reposition

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
    eta = 0.05*L*r**0.5 #resolution for Husimi
    k = 2*np.pi*np.fft.fftfreq(len(z),dz)
    #rescale wavenumber k to velocity v:
    hbar = 1
    v = k*(hbar/m)
    x_min, x_max = np.min(z), np.max(z)
    v_min, v_max = np.min(v), np.max(v)
    F = ND.Husimi_phase(chi,z,dz,L,eta)
    max_F = np.max(F)/2
    ax['upper right'].imshow(F,extent = (x_min,x_max,v_min,v_max),cmap = cm.coolwarm, norm = Normalize(0,max_F), aspect = (x_max-x_min)/(2*v_max))
    ax['upper right'].set_xlim(x_min,x_max)
    #ax['upper right'].set_ylim(-y01_max,y01_max) #[v_min,v_max])
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

    #Calculate rms velocity and position
    v_rms = np.sqrt(np.mean([star.v**2 for star in stars]))
    z_rms = np.sqrt(np.mean([star.x**2 for star in stars]))
    print(f"v_rms = {v_rms}")
    print(f"z_rms = {z_rms}")
    #v_rms = np.sqrt(np.sum([star.v**2 for star in stars])/Num_stars)

    K = 0.5 * v_rms**2
    print(f"K_avg = 0.5*m*v_rms^2 = {K} (m=1)")
    print(F"=> 2*K_avg = {2*K}")

    print(f"W_avg = {z_rms*Num_stars}")
    
    print("------------------")
    # Compute total KE of stars:
    K = 0
    for star in stars:
        dK = 0.5*sigma*star.v**2
        K += dK
    print(f"K_tot = {K}")
    #average KE:
    print(f"K_avg = {K/Num_stars}")

    # Compute Total Potential of stars:
    a_part = NB.acceleration(phi_part,L)
    W = 0
    for star in stars:
        g = NB.g(star,a_part,dz)

        dW = - sigma*star.x*g
        W += dW
    print(f"W_tot = {W}")
    print(f"W_avg = {W/Num_stars}")


    #Plot <v^2> vs |z|
    num_bins = int(np.floor(np.sqrt(Num_stars)))
    bins = np.zeros(num_bins)
    Delta = (L/2)/num_bins
    bins_counts = np.zeros(num_bins)
    for star in stars:
        i = int(np.abs(star.x)//Delta)
        bins[i] += star.v**2
        bins_counts[i] += 1
    v_rms_array = bins/bins_counts
    fig, ax = plt.subplots(1,2,figsize = (10,5))
    ax[0].set_title("Scatter plot of $v_{star}^2$ vs $|z_{star}|$")
    ax[0].scatter([np.abs(star.x) for star in stars], [star.v**2 for star in stars], s = 1)
    ax[0].set_xlabel("$|z|$")
    ax[0].set_ylabel("$v^2$")

    ax[1].set_title(f"RMS Velocity of Stars by histogrammed positions ({num_bins} bins)")
    ax[1].plot(np.linspace(0,L/2,len(v_rms_array)), np.sqrt(v_rms_array), 'b-', marker = "o")
    ax[1].set_xlabel("$|z|$")
    ax[1].set_ylabel("$\\sqrt{\\langle v^2 \\rangle}$")

    plt.show()

    # Plot Energies of the 5 Stars:
    fig,ax = plt.subplots(1,3, figsize = (15,5))
    for j in range(np.shape(K_5stars_Energies)[1]):
        KEs = K_5stars_Energies[:,j]
        Ws = W_5stars_Energies[:,j]
        ax[0].plot(KEs,marker=".",label = f"{j}-th Star")
        ax[0].set_title("Kinetic Energies")

        ax[1].plot(Ws,marker=".",label = f"{j}-th Star")
        ax[1].set_title("Potential Energies")

        ax[2].plot(KEs+Ws,marker=".",label = f"{j}-th Star")
        ax[2].set_title("Kinetic+Potential Energies")
        
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()
    
    W_totals = np.array([np.sum(W_Energies[i,:]) for i in range(np.shape(W_Energies)[0])])
    K_totals = np.array([np.sum(K_Energies[i,:]) for i in range(np.shape(K_Energies)[0])])
    Virial_ratios = np.abs(K_totals/W_totals)
    #print(Virial_ratios)
    indices = 99*np.array([0,1,2,4,8,16,32,64])
    
    fig,ax = plt.subplots(1,3,figsize = (15,5))
    ax[0].set_title("Total Energies over time")
    ax[0].plot(indices,W_totals,label = "$\\Sigma W$")
    ax[0].plot(indices,K_totals,label = "$\\Sigma K$")
    ax[0].legend()

    ax[1].plot(indices,K_totals+W_totals,label = "$\\Sigma E$")
    ax[1].legend()

    ax[2].set_title("Virial Ratio $|K/W|$ over time")
    ax[2].plot(indices, Virial_ratios, "b--", marker = "o")
    plt.show()

    Energies = K_Energies + W_Energies
    if Energies is None:
        pass
    else:
        #Plot each column
        #fig,ax = plt.subplots(np.shape(Energies)[1],1,figsize = (10,50))
        #plt.suptitle("Energy over Time of 5 random stars",fontsize = 20)
        indices = 99*[0,1,2,4,8,16,32,64,64.01]
        for i in range(np.shape(Energies)[0]-1):
            #print(len(Energies[i,:]))
            Delta_E = Energies[i+1,:]-Energies[i,:]
            Delta_E_avg = np.mean(Delta_E)
            plt.figure()
            plt.title(f"$\\Delta E_i$ vs $E_i$ @ i = {indices[i]}")
            plt.scatter(Energies[i,:],Delta_E,s = 5,marker = '.')
            plt.plot([np.min(Energies[i,:]),np.max(Energies[i,:])],[Delta_E_avg,Delta_E_avg],'r-',label = "Average")
            #plt.xlim(0,6500)
            plt.ylabel("$E_{"+f"{indices[i+1]}"+"}-E_{"+f"{indices[i]}"+"}$")
            plt.xlabel("$E_{"+f"{indices[i]}"+"}$")
            plt.legend()
            plt.show()    
            #for point in Energies[i,:]:
                #print(i, point)
             #   plt.scatter(indices[i],point,c = 'k', marker = '.')
        # plt.xlabel("Time (index)",fontsize = 15)
        # plt.ylabel("Energy")
        # plt.show()

    #Hist the final energies:
    plt.hist(Energies[-1,:],bins = 100)
    plt.show()

    ##########################################################
    # Split the distribution in half
    # Then add up to get rho vs |z|
    ##########################################################
    if Num_stars != 0: #Only do this if there are particles
        rho = rho_part


        #METHOD 1: Split across peak of distribution
        #Find center of distribution / max value and index:
        i = 0
        max_bool = False
        while max_bool == False:
            for j in range(len(rho)):
                if rho[j] > rho[i]: #if you come across an index j that points to a larger value..
                    #then set i equal to j
                    i = j 
                    #break
                else:
                    max_index = i
                    max_bool = True

        # max_rho = rho[max_index]
        # print(max_rho,max_index,z[i])


        i = max_index
        z = z-z[i]
        z_left = z[0:i]
        z_right = z[i:]
        rho_left = rho[0:i]
        rho_right = rho[i:]

        #rho_avgd = (rho_left[len(rho_left)-len(rho_right):][::-1]+rho_right)/2
        #rho_avgd = np.append(rho_avgd, rho_left[0:len(rho_left)-len(rho_right)][::-1])
        fig = plt.figure()
        plt.title("Density of Particles Split in Half")
        plt.plot(z_right,rho_right)
        plt.plot(z_left,rho_left)
        plt.plot(z[i],rho[i], "ro", label = "Peak of Distribution")
        plt.legend()
        plt.show()

        fig = plt.figure()
        plt.title("Density of Particles Split in Half")
        plt.plot(z[N//2:],rho[N//2:])
        plt.plot(z[0:N//2],rho[0:N//2])
        plt.plot(z[N//2],rho[N//2],"bo", label = "Centroid of Distribution")
        plt.legend()
        plt.show()


        #Other method to accumulate left and right sides:
        # for star in stars:
        #     star.x = star.x - z[i] #shift
        #     star.reposition(L) #reposition

        # grid_counts = NB.grid_count(stars,L,z)
        # rho_part = (grid_counts/dz)*sigma 
        # #Add the density from the FDM
        # rho_FDM = np.absolute(chi)**2 
        # rho = rho_FDM + rho_part

        #Find center of distribution / max value and index:
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

        #METHOD 2: Split across z = 0 (i.e: z[N//2])
        rho_left = rho[0:N//2]
        rho_right = rho[N//2:]
        rho_whole = rho_left[::-1] + rho_right

        z_left = z[0:N//2]
        z_right = z[N//2:]

        fig,ax = plt.subplots(1,2,figsize = (10,4))
        plt.suptitle("Combined Left and Right halves of Distribution")
        ax[0].plot(z_right,rho_whole,'--')
        ax[0].set_xlabel("$|z|$")
        ax[0].set_ylabel("$|rho|$")

        ax[1].plot(np.log(z_right),np.log(rho_whole))
        ax[1].set_xlabel("$log|z|$")
        ax[1].set_ylabel("$log|rho|$")
        plt.show()

        ###################################################
        # Curve Fitting
        ###################################################

        # Note: Before fitting, we have to shift the z_array up a tiny amount
        # ...b/c if there is a z = 0.0, there will be a divide by zero error

        #Just skip the very first element:
        z_right = z_right[2:]#+1E-10
        rho_whole = rho_whole[2:]


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

        cols=0
        try:
            guess_params = [1,1]
            popt1,pcov1 = opt.curve_fit(fit_func,z_right,rho_whole,guess_params,maxfev = 5000)
            print("Check")
            cols += 1 
            try: 
                guess_params = [1,1,0]#np.append(popt,0)
                popt2,pcov2 = opt.curve_fit(new_fit_func,z_right,rho_whole,guess_params)
                print("Check")
                cols += 1 
                try: 
                    guess_params = [1,1,0]#np.append(popt,0)
                    popt3,pcov3 = opt.curve_fit(new_new_fit_func,z_right,rho_whole,guess_params)
                    print("Check")
                    cols += 1 
                    try:
                        guess_params = np.append(popt3,2)# [1,1,0,2]
                        popt4,pcov4 = opt.curve_fit(new_new_new_fit_func,z_right,rho_whole,guess_params,maxfev = 5000)
                        print("Check")
                        cols += 1 
                    except:
                        pass            
                except:
                    pass
            except:
                pass
        except:
            pass
            
        if cols ==0:
            cols = 1

        fig,ax = plt.subplots(2,cols,figsize = (30,10))
        plt.suptitle("Density vs |z| with Curve fit",fontsize = 25)
        
        fit_rho = fit_func(z_right,*popt1)
        ax[0,0].plot(z_right,rho_whole)
        ax[0,0].plot(z_right,fit_rho,'r--',label="Curve Fit")
        ax[0,0].set_xlim(-0.1,1.1)#L/2)
        ax[0,0].text(L/8,max(rho_whole)*3/4, "$f(|z|) = \\frac{a_0}{|z|(|z|+a_1)^2}$",fontsize = 30)
        ax[0,0].text(L/8,max(rho_whole)*1/2, f"$a_0 = {popt1[0]}$",fontsize = 15)
        ax[0,0].text(L/8,0.85*max(rho_whole)*1/2, f"$a_1 = {popt1[1]}$",fontsize = 15)
        
        ax[0,0].legend(fontsize = 25)

        residuals = fit_rho-rho_whole
        resid_y_max = np.max(residuals)
        ax[1,0].plot(z_right,residuals,"r.--")
        ax[1,0].set_xlim(-0.1,1.1)#L/2)
        ax[1,0].set_ylim((-resid_y_max,resid_y_max))
        #ax[1,0].legend()

        chi2 = 0
        for i in range(len(residuals)):
            chi2 += (residuals[i])**2 / fit_rho[i]
        ax[1,0].text(L/4, 0.8*np.max(residuals), f"$chi^2$ = {chi2}")

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

