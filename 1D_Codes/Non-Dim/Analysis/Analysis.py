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
My_Package_PATH = "/home/boris/Documents/Research/Coding"
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

    L = 2
    N = 10**3
    z = np.linspace(-L/2,L/2,N)
    dz = z[1]-z[0]

    if Num_bosons == 0:
        folder = "ParticlesOnly_Snapshots"
        stars_x = np.loadtxt(folder+"/"+f"StarsOnly_Pos.csv", dtype = float, delimiter=",")
        stars_v = np.loadtxt(folder+"/"+f"StarsOnly_Vel.csv", dtype = float, delimiter=",")
        Energies = np.loadtxt(folder+"/"+"Energies.csv", dtype = float,delimiter = ",")
        #chi = np.loadtxt(folder+"/"+f"Chi.csv", dtype = complex, delimiter=",")
        chi = np.zeros_like(z)
    elif Num_stars == 0:
        folder = f"OnlyFuzzyMass{m}_Snapshots"
        chi = np.loadtxt(folder+"/"+f"FuzzyOnlyChi_m{m}.csv", dtype = complex, delimiter=",")
        Energies = None
    elif Num_bosons!=0 and Num_stars !=0:
        folder = f"FuzzyMass{m}_Snapshots"
        stars_x = np.loadtxt(folder+"/"+f"Stars_Pos_m{m}.csv", dtype = float, delimiter=",")
        stars_v = np.loadtxt(folder+"/"+f"Stars_Vel_m{m}.csv", dtype = float, delimiter=",")
        chi = np.loadtxt(folder+"/"+f"Chi_m{m}.csv", dtype = complex, delimiter=",")
        Energies = np.loadtxt(folder+"/"+f"Energies_m{m}.csv", dtype = float,delimiter = ",")

    # stars_x = np.loadtxt(f"Stars_Pos.csv", dtype = float, delimiter=",")
    # stars_v = np.loadtxt(f"Stars_Vel.csv", dtype = float, delimiter=",")
    # chi = np.loadtxt(f"Chi.csv", dtype = complex, delimiter=",")





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
    ax['upper right'].imshow(F,extent = (x_min,x_max,v_min,v_max),cmap = cm.hot, norm = Normalize(0,max_F), aspect = (x_max-x_min)/(2*v_max))
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
        
        ax['lower right'].scatter([stars_x[i] for i in range(5)],[stars_v[i] for i in range(5)], c = 'black', s = 50, marker = 'o')
        
        ax['lower right'].scatter(centroid_z,0,s = 100,c = "r",marker = "o")
    
    plt.show()

    if Energies is None:
        pass
    else:
        #Plot each column
        #fig,ax = plt.subplots(np.shape(Energies)[1],1,figsize = (10,50))
        #plt.suptitle("Energy over Time of 5 random stars",fontsize = 20)
        fig = plt.plot(figsize = (15,15))
        plt.title("Energy over Time of 5 random stars",fontsize = 15)
        for i in range(np.shape(Energies)[1]):
            plt.plot(Energies[:,i])
        plt.xlabel("Time (index)",fontsize = 15)
        plt.ylabel("Energy")
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

        print(z_right[0])

        fig = plt.figure()
        plt.title("Combined Left and Right halves of Distribution")
        # plt.plot(z_right,rho_right)
        # plt.plot(z_left,rho_left)
        plt.plot(z_right,rho_whole,'--')
        #plt.plot(z[N//2],rho_whole[N//2], "ro")
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

        try:
            guess_params = [1,1]
            popt,pcov = opt.curve_fit(fit_func,z_right,rho_whole,guess_params)
            
            fig,ax = plt.subplots(2,1,figsize = (10,15))
            plt.suptitle("Density vs |z| with Curve fit")
            fit_rho = fit_func(z_right,*popt)
            ax[0].plot(z_right,rho_whole)
            ax[0].plot(z_right,fit_rho,'r--',label="Curve Fit")
            ax[0].set_xlim(-0.1,1.1)#L/2)
            ax[0].text(L/4,max(rho_whole)*3/4, "$f(|z|) = \\frac{a_0}{|z|(|z|+a_1)^2}$",fontsize = 30)
            ax[0].text(L/8,max(rho_whole)*1/2, f"$a_0 = {popt[0]}$",fontsize = 15)
            ax[0].text(L/8,0.85*max(rho_whole)*1/2, f"$a_1 = {popt[1]}$",fontsize = 15)
            
            ax[0].legend(fontsize = 30)

            residuals = fit_rho-rho_whole
            ax[1].plot(z_right,residuals,"r.--")
            ax[1].set_xlim(-0.1,1.1)#L/2)
            #ax[1].legend()

            plt.show()
            chi2 = 0
            for i in range(len(residuals)):
                chi2 += (residuals[i])**2 / fit_rho[i]
            print(f"chi^2 = {chi2}")
        except:
            pass

        #re-try
        def new_fit_func(z,*pars):
            a0,a1,a2 = pars
            og = a0/(z*(z+a1)**2)
            correction = -a2/z
            return og - correction

        try:
            guess_params = [1,1,0]#np.append(popt,0)
            popt,pcov = opt.curve_fit(new_fit_func,z_right,rho_whole,guess_params)
            
            fig,ax = plt.subplots(2,1,figsize = (10,15))
            plt.suptitle("Density vs |z| with Curve fit")
            fit_rho = new_fit_func(z_right,*popt)
            ax[0].plot(z_right,rho_whole)
            ax[0].plot(z_right,fit_rho,'r--',label="Curve Fit")
            ax[0].set_xlim(-0.1,1.1)#L/2)
            ax[0].text(L/4,max(rho_whole)*3/4, "$f(|z|) = \\frac{a_0}{|z|(|z|+a_1)^2}-\\frac{a_2}{|z|}$",fontsize = 30)
            ax[0].text(L/8,max(rho_whole)*1/2, f"$a_0 = {popt[0]}$",fontsize = 15)
            ax[0].text(L/8,0.85*max(rho_whole)*1/2, f"$a_1 = {popt[1]}$",fontsize = 15)
            ax[0].text(L/8,0.70*max(rho_whole)*1/2, f"$a_2 = {popt[2]}$",fontsize = 15)
            
            ax[0].legend(fontsize = 30)

            residuals = fit_rho-rho_whole
            ax[1].plot(z_right,residuals,"r.--")
            ax[1].set_xlim(-0.1,1.1)#L/2)
            #ax[1].legend()

            plt.show()
            chi2 = 0
            for i in range(len(residuals)):
                chi2 += (residuals[i])**2 / fit_rho[i]
            print(f"chi^2 = {chi2}")
        except:
            pass

        def new_new_fit_func(z,*pars):
            a0,a1,a2 = pars
            og = a0/((z**a2) * (z+a1)**2)
            return og

        try: 
            guess_params = [1,1,0]#np.append(popt,0)
            popt,pcov = opt.curve_fit(new_new_fit_func,z_right,rho_whole,guess_params)
            
            fig,ax = plt.subplots(2,1,figsize = (10,15))
            plt.suptitle("Density vs |z| with Curve fit")
            fit_rho = new_new_fit_func(z_right,*popt)
            ax[0].plot(z_right,rho_whole)
            ax[0].plot(z_right,fit_rho,'r--',label="Curve Fit")
            ax[0].set_xlim(-0.1,1.1)#L/2)
            ax[0].text(L/4,max(rho_whole)*3/4, "$f(|z|) = \\frac{a_0}{|z|^{a_2}(|z|+a_1)^2}$",fontsize = 30)
            ax[0].text(L/8,max(rho_whole)*1/2, f"$a_0 = {popt[0]}$",fontsize = 15)
            ax[0].text(L/8,0.85*max(rho_whole)*1/2, f"$a_1 = {popt[1]}$",fontsize = 15)
            ax[0].text(L/8,0.70*max(rho_whole)*1/2, f"$a_2 = {popt[2]}$",fontsize = 15)
            
            ax[0].legend(fontsize = 30)

            residuals = fit_rho-rho_whole
            ax[1].plot(z_right,residuals,"r.--")
            ax[1].set_xlim(-0.1,1.1)#L/2)
            #ax[1].legend()

            plt.show()
            chi2 = 0
            for i in range(len(residuals)):
                chi2 += (residuals[i])**2 / fit_rho[i]
            print(f"chi^2 = {chi2}")
        except:
            pass

        def new_new_new_fit_func(z,*pars):
            a0,a1,a2,a3 = pars
            og = a0/((z**a2) * (z+a1)**a3)
            return og

        try:
            guess_params = np.append(popt,2)# [1,1,0,2]
            popt,pcov = opt.curve_fit(new_new_new_fit_func,z_right,rho_whole,guess_params,maxfev = 5000)
            
            fig,ax = plt.subplots(2,1,figsize = (10,15))
            plt.suptitle("Density vs |z| with Curve fit")
            fit_rho = new_new_new_fit_func(z_right,*popt)
            ax[0].plot(z_right,rho_whole)
            ax[0].plot(z_right,fit_rho,'r--',label="Curve Fit")
            ax[0].set_xlim(-0.1,1.1)#L/2)
            ax[0].text(L/4,max(rho_whole)*3/4, "$f(|z|) = \\frac{a_0}{|z|^{a_2}(|z|+a_1)^{a_3}}$",fontsize = 30)
            ax[0].text(L/8,max(rho_whole)*1/2, f"$a_0 = {popt[0]}$",fontsize = 15)
            ax[0].text(L/8,0.85*max(rho_whole)*1/2, f"$a_1 = {popt[1]}$",fontsize = 15)
            ax[0].text(L/8,0.70*max(rho_whole)*1/2, f"$a_2 = {popt[2]}$",fontsize = 15)
            ax[0].text(L/8,0.55*max(rho_whole)*1/2, f"$a_3 = {popt[3]}$",fontsize = 15)
            
            ax[0].legend(fontsize = 30)

            residuals = fit_rho-rho_whole
            ax[1].plot(z_right,residuals,"r.--")
            ax[1].set_xlim(-0.1,1.1)#L/2)
            #ax[1].legend()

            plt.show()
            chi2 = 0
            for i in range(len(residuals)):
                chi2 += (residuals[i])**2 / fit_rho[i]
            print(f"chi^2 = {chi2}")
        except: 
            pass


        ###################################################
        plt.figure()
        plt.title("Gravitational Potential in the Box")
        fit_rho = np.append(fit_rho[::-1],fit_rho)/ 2 #have to divide by 2 becasue we are double counting on the grid
        fit_phi = GF.fourier_potentialV2(fit_rho,L) 
        fit_z = np.linspace(-L/2,L/2,len(fit_phi))
        plt.plot(fit_z,fit_phi,label = "Analytic Model")
        rho = np.append(rho_left,rho_right)
        plt.plot(z,GF.fourier_potentialV2(rho,L), label = "Exact NBody Potential")
        plt.legend()
        plt.show()

