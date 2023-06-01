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

import OneD.FDM as FDM
import OneD.NBody as NB
import OneD.Global as GF

def plot_centroids(indices,centroids):
    
    indices_new = [float(x) for x in indices][:len(centroids)]
        
    plt.figure()
    plt.title("Position Centroid over time")
    plt.plot(indices_new,centroids[:,0],'bo-')
    #plt.scatter(centroids[:,0],centroids[:,1])
    plt.ylim(-1,1)
    plt.show()
    
    plt.figure()
    plt.title("Velocity Centroid over time")
    plt.plot(indices_new,centroids[:,1],'ro-')
    plt.show()

def rms_stuff(sigma,stars,phi_part,L,z,dz,type = 'Periodic'):
    Num_stars = len(stars)

    grid_counts = NB.grid_count(stars,L,z)
    centroid_z = 0
    for j in range(len(grid_counts)):
        centroid_z += z[j]*grid_counts[j]
    centroid_z = centroid_z / Num_stars
    
    for star in stars:
        star.x = star.x - centroid_z #shift
        star.reposition(L) #reposition

    v_rms = np.sqrt(np.mean([star.v**2 for star in stars]))
    z_rms = np.sqrt(np.mean([star.x**2 for star in stars]))
    print(f"v_rms = {v_rms}")
    print(f"z_rms = {z_rms}")
    #v_rms = np.sqrt(np.sum([star.v**2 for star in stars])/Num_stars)

    K = 0.5 * sigma * v_rms**2
    print(f"K_avg = 0.5*m*v_rms^2 = {K} (m={sigma})")
    print(F"=> 2*K_avg = {2*K}")

    W = z_rms*Num_stars
    print(f"W_avg = {W}")

    print("---------Now Different Routine---------")
    ### Now differnet routine:
    # Compute total KE of stars:
    K = 0
    for star in stars:
        dK = 0.5*sigma*star.v**2
        K += dK
    print(f"K_tot = {K}")
    #average KE:
    print(f"K_avg = {K/Num_stars}")

    # Compute Total Potential of stars:
    a_part = NB.acceleration(phi_part,L,type = type)
    W = 0
    for star in stars:
        g = NB.g(star,a_part,dz)

        dW = - sigma*star.x*g
        W += dW
    print(f"W_tot = {W}")
    print(f"W_avg = {W/Num_stars}")

    return z_rms, v_rms 


def rms_plots(indices, z_rms_s,v_rms_s):
    #i = 99*np.array([0,1,2,4,8,16,32,64])
    
    fig,ax = plt.subplots(1,2,figsize = (12,5))
    plt.suptitle("RMS values over time")
    
    ax[0].plot(indices, z_rms_s[:len(indices)], "o-")
    ax[0].set_xlabel("Time [index]")
    ax[0].set_title("$z_{rms}$")
    ax[0].set_ylim(0,1)

    ax[1].plot(indices, v_rms_s[:len(indices)], "o-")
    ax[1].set_xlabel("Time [index]")
    ax[1].set_title("$v_{rms}$")
    ax[1].set_ylim(0,1)
    
    plt.show()
    
def v_distribution(stars,L):
    Num_stars= len(stars)
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

def select_stars_plots(z,K_5stars_Energies,W_5stars_Energies):
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
    
    W_totals = 0.5 * np.array([np.sum(W_5stars_Energies[i,:]) for i in range(np.shape(W_5stars_Energies)[0])])
    K_totals = np.array([np.sum(K_5stars_Energies[i,:]) for i in range(np.shape(K_5stars_Energies)[0])])
    Virial_ratios = np.abs(K_totals/W_totals)
    #print(Virial_ratios)
    indices = z#99*np.array([0,1,2,4,8,16,32,64])
    
    fig,ax = plt.subplots(1,4,figsize = (20,5))
    plt.suptitle("Total Energy Plots for the 5 stars")
    ax[0].set_title("Total Potential Energy over time")
    ax[0].plot(W_totals,label = "$\\Sigma W$")
    ax[0].legend()

    ax[1].set_title("Total Kinetic Energy over time")
    ax[1].plot(K_totals,label = "$\\Sigma K$")
    ax[1].legend()

    ax[2].set_title("Total Energy K+W over time")
    ax[2].plot(K_totals+W_totals,label = "$\\Sigma E$")
    ax[2].legend()

    ax[3].set_title("Virial Ratio $|K/W|$ over time")
    ax[3].plot(Virial_ratios, "b--",marker = ".")
    plt.show()

def plot_Energies(time,Ks,Ws,variable_mass=None):
    
    if variable_mass[0] == 'False':
        # fig, ax = plt.subplots(2,1, figsize = (10,8), sharex=True, gridspec_kw = {'height_ratios': [2.5,1]})
    
        K = Ks# - np.mean(Ks)
        W = Ws# - np.mean(Ws)
        E = Ks/np.abs(Ws) #+Ws 
        E = E - np.mean(E)

        RMS_amplitude = np.sqrt(np.mean(E**2)) #np.sqrt(np.mean(E**2))
        Max_amplitude = np.max(E)
       
        # ax[0].set_title("Variation from Mean Total Energies in Particles over Time", fontdict={'fontsize' : 15})
        # ax[0].plot(time,K,label = "$\\delta K$ Kinetic Energy")
        # ax[0].plot(time,W,label = "$\\delta W$ Potential Energy")
        # ax[0].plot(time,E, label = "$\\delta (K+W)$ Total Energy")
        # ax[0].legend(loc='upper right')
        # ax[0].set_ylabel("Energy (code units)")
        # ax[0].set_xlabel("Time (code units)")

        # ax[1].set_title("Ratio of Kinetic to Potential Energy $\\frac{K}{|W|}$", fontdict={'fontsize' : 15})
        # ax[1].plot(time,Ks/np.abs(Ws)) #, label = "Virial Ratio $\\frac{K}{|W|}$")
        # ax[1].set_xlabel("Time (code units)")
    
        # ax[0].grid(True)
        # ax[1].grid(True)

        # fig.subplots_adjust(hspace = 0.3)
        # plt.show()

    else:
        fig, ax = plt.subplots(2,2, figsize = (20,10), sharex=True, gridspec_kw = {'height_ratios': [2.5,1]})

        V1 = Ks[:,0]/np.abs(Ws[:,0])
        V2 = Ks[:,1]/np.abs(Ws[:,1])
        
        K1 = Ks[:,0] - np.mean(Ks[:,0])
        W1 = Ws[:,0] - np.mean(Ws[:,0])
        E1 = K1+W1
        E1 = E1 - np.mean(E1)

        K2 = Ks[:,1] - np.mean(Ks[:,1])
        W2 = Ws[:,1] - np.mean(Ws[:,1])
        E2 = K2+W2
        E2 = E2 - np.mean(E2)

        RMS_amplitude = np.sqrt(np.mean(E1**2))
        Max_amplitude = np.max(E1)
        
        plt.suptitle("Variation from Mean Total Energies in Particles over Time", fontsize = 15) #fontdict={'fontsize' : 15})
        
        ax[0,0].plot(time,K1,label = "$\\delta K$ Kinetic Energy")
        ax[0,0].plot(time,W1,label = "$\\delta W$ Potential Energy")
        ax[0,0].plot(time,E1,label = "$\\delta (K+W)$ Total Energy")
        ax[0,0].set_title("Heavy Particles")

        ax[0,1].plot(time,K2,label = "$\\delta K$ Kinetic Energy")
        ax[0,1].plot(time,W2,label = "$\\delta W$ Potential Energy")
        ax[0,1].plot(time,E2,label = "$\\delta (K+W)$ Total Energy")
        ax[0,1].set_title("Light Particles")
        
        ax[0,0].legend(loc='upper right')
        ax[0,1].legend(loc='upper right')

        ax[1,0].set_title("Heavy Particles Ratio of Kinetic to Potential Energy $\\frac{K}{|W|}$", fontdict={'fontsize' : 15})
        ax[1,0].plot(time,V1)
        ax[1,1].set_title("Light Particles Ratio of Kinetic to Potential Energy $\\frac{K}{|W|}$", fontdict={'fontsize' : 15})
        ax[1,1].plot(time,V2)
        
        
        #ax[1,0].set_ylim(0,1)
        #ax[1,1].set_ylim(0,1)

        ax[0,0].set_ylabel("Energy (code units)")
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

    return RMS_amplitude, Max_amplitude

def plot_Freqs(time,Ks,Ws):
    

    dtau = time[1]-time[0]
    k = 2*np.pi*np.fft.rfftfreq(len(Ks),dtau)
    
    Ks = np.fft.rfft(Ks-np.mean(Ks))
    Ws = np.fft.rfft(Ws-np.mean(Ws))
    

    K = np.abs(Ks)**2
    W = np.abs(Ws)**2
    #E = K+W
    E=K

    fig, ax = plt.subplots(1,1, figsize = (15,5), sharex = True)
    fig.suptitle("Power Spectrum for Oscillations in Kinetic Energy", fontsize = 15)
    ax.plot(k**(1/3), E,"-")
    ax.set_xlabel("Cube Root Frequency $\\sqrt[3]{k}$",fontsize=12)
    ax.set_ylabel("Power", fontsize=12)
    
    plt.grid(True)
    plt.yscale("log")
    plt.show()



def all_stars_plots(indices,K_Energies,W_Energies, variable_mass = [False]):
    W_totals = 0.5 * np.array([np.sum(W_Energies[i,:]) for i in range(np.shape(W_Energies)[0])])
    K_totals = np.array([np.sum(K_Energies[i,:]) for i in range(np.shape(K_Energies)[0])])
    
    
    # C = K_totals+W_totals
    # W_totals -= C
    Virial_ratios = np.abs(K_totals/W_totals)
    #print(Virial_ratios)
    
    #set the scale:
    Dy_1 = np.max(K_totals)-np.min(K_totals)
    Dy_2 = np.max(W_totals)-np.min(W_totals)
    Dy = np.max([Dy_1,Dy_2])
    
    fig,ax = plt.subplots(1,4,figsize = (15,5))
    plt.suptitle("Energy Plots for Every Star, at Snapshot times/indices")
    
    print(variable_mass)
    #check whether there is variable mass:
    if variable_mass[0] == 'True':
        print(variable_mass[0])
        fraction = variable_mass[1]
        Num_stars = len(K_Energies[0,:])
        print(Num_stars)
        num_to_change = int(np.floor(fraction*Num_stars))
        print(num_to_change)
        W_totals = 0.5 * np.array([np.sum(W_Energies[i,:num_to_change]) for i in range(np.shape(W_Energies)[0])])
        K_totals = np.array([np.sum(K_Energies[i,:num_to_change]) for i in range(np.shape(K_Energies)[0])])
        W_totals2 = 0.5 * np.array([np.sum(W_Energies[i,num_to_change:]) for i in range(np.shape(W_Energies)[0])])
        K_totals2 = np.array([np.sum(K_Energies[i,num_to_change:]) for i in range(np.shape(K_Energies)[0])])
        
        ax[0].plot(indices,W_totals2,"b--", marker = "o",label = "Light")
        ax[1].plot(indices,K_totals2,"b--", marker = "o",label = "Light")
        ax[2].plot(indices,K_totals2+W_totals2,"b--", marker = "o",label = "$\\Sigma E$")
        
        ax[2].plot(indices,(K_totals+K_totals2+W_totals+ W_totals2)/2,"k--", marker = "o",label = "(Heavy + Light) /2")
        y_tot_avg = np.mean((K_totals+K_totals2+W_totals+ W_totals2)/2)
    
        Virial1 = np.abs(K_totals/W_totals)
        Virial2 = np.abs(K_totals2/W_totals2)
        ax[3].plot(indices,Virial1, "r--", label = "Heavy")
        ax[3].plot(indices,Virial2, "b--", label = "Light")
    else:
        y_tot_avg = np.mean((K_totals+W_totals))
    ax[0].set_title("Potential Energy over time")
    ax[0].plot(indices,W_totals,"r--", marker = "o", label = "Heavy")
    ax[0].legend()
    
    ax[1].set_title("Kinetic Energy over time")
    ax[1].plot(indices,K_totals,"r--", marker = "o",label = "Heavy")
    ax[1].legend()
    
    if variable_mass[0] == 'True':
        y_avg = np.mean([np.mean(W_totals+K_totals),np.mean(W_totals2+K_totals2)])
    else: 
        y_avg = y_tot_avg
    y_min = y_avg - Dy
    y_max = y_avg + Dy
    ax[2].set_title("Total Energy K+W over time")
    ax[2].plot(indices,K_totals+W_totals,"r--", marker = "o",label = "Heavy")
    #ax[2].set_ylim(y_min,y_max)
    ax[2].legend()

    ax[3].set_title("Virial Ratio $|K/W|$ over time")
    ax[3].plot(indices, Virial_ratios, "b-", marker = "o")
    ax[3].legend()
    plt.show()

def scatter_deltaE(Energies_i, Energies_f, variable_mass, Num_bosons, r = None):
    plt.figure()
    if variable_mass[0]=='True':
        fraction = variable_mass[1]
        Num_stars = len(Energies_i)

        num_to_change = int(np.floor(fraction*Num_stars))
        # plt.scatter(Energies[0,:num_to_change],Energies[len(indices)-1,:num_to_change], c = "red", marker = ".", s = 1,label = "Heavy")
        # plt.scatter(Energies[0,num_to_change:],Energies[len(indices)-1,num_to_change:], c = "blue", marker = ".", s = 1, label = "Light")
        plt.scatter(Energies_i[:num_to_change],Energies_f[:num_to_change], c = "red", marker = ".", s = 1,label = "Heavy")
        plt.scatter(Energies_i[num_to_change:],Energies_f[num_to_change:], c = "blue", marker = ".", s = 1, label = "Light")

    else:
        xy = np.vstack([Energies_i,Energies_f])
        z = gaussian_kde(xy)(xy)
        plt.scatter(Energies_i[:],Energies_f[:], c=z,marker = ".", s = 1)   
        
    x_0 = np.min(Energies_i)
    x_1 = np.max(Energies_i)
    plt.plot([x_0,x_1],[x_0,x_1], "r-", label = "$y=x$")
    plt.title("Scattered $E_{final}$ vs $E_{initial}$ (of Stars)")
    plt.xlabel("$E_{initial}$")
    plt.ylabel("$E_{final}$")
    if Num_bosons!=0:
        plt.text((x_0+x_1)/2,((x_0+x_1)),f"r = {r}")
    plt.colorbar()
    plt.legend()
    plt.show()

    deltaE = np.sum(Energies_f - Energies_i)
    print(f"Net change in Energy (of all stars)= {deltaE}")
        

    if variable_mass[0]=='True':
        fig, ax = plt.subplots(1,2, figsize =(12,6))
    
        #fraction = variable_mass[1]
        #num_to_change = int(np.floor(fraction*Num_stars))
        #Energies = W_Energies + K_Energies
        xy0 = np.vstack([Energies_i[:num_to_change],Energies_f[:num_to_change]])
        z0 = gaussian_kde(xy0)(xy0)
        ax[0].scatter(Energies_i[:num_to_change],Energies_f[:num_to_change], c = z0, marker = ".", s = 10,label = "Heavy")
        x_0 = np.min(Energies_i[:num_to_change])
        x_1 = np.max(Energies_i[:num_to_change])
        ax[0].plot([x_0,x_1],[x_0,x_1], "k-", label = "$y=x$")
        # ax[0].set_xlim(-1e-7, 5e-6)
        # ax[0].set_ylim(-1e-7, 5e-6)
        ax[0].set_title("Heavier Particles")
        

        xy1 = np.vstack([Energies_i[num_to_change:],Energies_f[num_to_change:]])
        z1 = gaussian_kde(xy1)(xy1)
        ax[1].scatter(Energies_i[num_to_change:],Energies_f[num_to_change:], c = z1, marker = ".", s = 1, label = "Light")
        x_0 = np.min(Energies_i[num_to_change:])
        x_1 = np.max(Energies_i[num_to_change:])
        ax[1].plot([x_0,x_1],[x_0,x_1], "k-", label = "$y=x$")
        ax[1].set_title("Lighter Particles")
        ax[1].set_xlabel("$E_{initial}$", fontsize = 14)
        ax[0].set_xlabel("$E_{initial}$", fontsize = 14)
        ax[0].set_ylabel("$E_{final}$", fontsize = 14)
        
        sigma1 = variable_mass[2]
        sigma2 = variable_mass[3]
        plt.suptitle("Scattered $E_{final}$ vs $E_{initial}$ (of Stars)", fontsize = 20)
        plt.title("$M_{heavy}/m_{light}=$"+f"{sigma1/sigma2}")
        plt.legend()
        plt.show()

        deltaE = np.sum(Energies_f[num_to_change:] - Energies_i[num_to_change:])
        print(f"Net change in Energy = {deltaE}")

    return deltaE

def scatter_deltaE_frac(Energies_i, Energies_f, variable_mass, Num_bosons, r = None):
    
    x = Energies_i
    y = (Energies_f-Energies_i)/Energies_i

    if variable_mass[0]=='True':
        fraction = variable_mass[1]
        Num_stars = len(Energies_i)
        num_to_change = int(np.floor(fraction*Num_stars))
        
        fig, ax = plt.subplots(1,2, figsize =(12,6))
    
        #fraction = variable_mass[1]
        #num_to_change = int(np.floor(fraction*Num_stars))
        #Energies = W_Energies + K_Energies
        xy0 = np.vstack([x[:num_to_change],y[:num_to_change]])
        z0 = gaussian_kde(xy0)(xy0)
        z0_plot = ax[0].scatter(x[:num_to_change],y[:num_to_change], c = z0, marker = ".", s = 10,label = "Heavy")
        x_0 = np.min(x[:num_to_change])
        x_1 = np.max(Energies_i[:num_to_change])
        ax[0].plot([x_0,x_1],[0,0], "k-", label = "$y=0$")
        ax[0].set_title("Heavier Particles")
        


        xy1 = np.vstack([x[num_to_change:],y[num_to_change:]])
        z1 = gaussian_kde(xy1)(xy1)
        z1_plot = ax[1].scatter(x[num_to_change:],y[num_to_change:], c = z1, marker = ".", s = 1, label = "Light")
        x_0 = np.min(x[num_to_change:])
        x_1 = np.max(x[num_to_change:])
        ax[1].plot([x_0,x_1],[0,0], "k-", label = "$y=0$")
        ax[1].set_title("Lighter Particles")
        
        ax[1].set_xlabel("$E_{initial}$", fontsize = 14)
        ax[0].set_xlabel("$E_{initial}$", fontsize = 14)
        ax[0].set_ylabel("$\\Delta E / E_{initial}$", fontsize = 14)
        

        plt.colorbar(z0_plot,ax=ax[0])
        plt.colorbar(z1_plot,ax=ax[1])
        
        sigma1 = variable_mass[2]
        sigma2 = variable_mass[3]
        
        #plt.text("$M_{heavy}/m_{light}=$"+f"{sigma1/sigma2}",0,0)
    
        deltaE = np.sum(Energies_f[num_to_change:] - Energies_i[num_to_change:])
        print(f"Net change in Energy = {deltaE}")
    else:
        plt.figure()
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        plt.scatter(x,y, c=z,marker = ".", s = 1)   

        x_0 = np.min(x)
        x_1 = np.max(x)
        plt.plot([x_0,x_1],[0,0], "k-", label = "$y=0$")
        
        plt.xlabel("$E_{initial}$", fontsize = 14)
        plt.ylabel("$\\Delta E / E_{initial}$", fontsize = 14)

        deltaE = np.sum(Energies_f[:] - Energies_i[:])
        print(f"Net change in Energy = {deltaE}")

        plt.colorbar()
    
    plt.suptitle("Scattered $\\Delta E / E_{initial}$ vs $E_{initial}$ (Particles)", fontsize = 20)
    plt.legend()
    plt.show()

    return deltaE

def plot_DeltaE(K_Energies,W_Energies):
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

        
def rho_distribution(z,rho_part):
    ##########################################################
    # Split the distribution in half
    # Then add up to get rho vs |z|
    ##########################################################
    rho = rho_part
    N = len(z)

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

    i = max_index
    #z = z-z[i]

    z_left = z[0:i]
    z_right = z[i:]
    rho_left = rho[0:i]
    rho_right = rho[i:]

    fig = plt.figure()
    plt.title("Density of Particles Split in Half")
    plt.plot(z_right,rho_right)
    plt.plot(z_left,rho_left)
    plt.plot(z[i],rho[i], "ro", label = "Peak of Distribution")
    plt.legend()
    plt.show()

    rho_left = rho_left[::-1]
    if len(rho_left) >= len(rho_right):
        rho_whole = 0.5*(rho_left[0:len(rho_right)]+rho_right) #np.append(0.5*(rho_left[0:len(rho_right)]+rho_right) , rho_left[len(rho_right):])
    elif len(rho_right)>len(rho_left):
        rho_whole = 0.5*(rho_left+rho_right[0:len(rho_left)]) #np.append(0.5*(rho_left+rho_right[0:len(rho_left)]) , rho_left[len(rho_left):])
    z_whole = np.linspace(0.001,1,len(rho_whole))
    
    fig,ax = plt.subplots(1,2,figsize = (10,5))
    
    ax[0].plot(z_whole,rho_whole)
    ax[0].set_title("Particles Density Profile")
    ax[0].set_xlabel("$|z|$")
    ax[0].set_ylabel("$|\\rho|$")
    
    ax[1].plot(np.log(z_whole),np.log(rho_whole))
    ax[1].set_title("Particles Density Profile")
    ax[1].set_xlabel("$ln|z|$")
    ax[1].set_ylabel("$ln|\\rho|$")
    
    plt.show()
    
    return z_whole, rho_whole #z_left,z_right,rho_left,rho_right
