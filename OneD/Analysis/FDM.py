import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
import os
import subprocess
import cv2 
from PIL import Image 
import scipy.optimize as opt

import OneD.FDM as FDM
import OneD.NBody as NB
import OneD.Global as GF



def plot_Energies(time,Ks,Ws):
    K = Ks# - np.mean(Ks)
    W = Ws# - np.mean(Ws)
    E = Ks/np.abs(Ws) #+Ws 
    E = E - np.mean(E)

    RMS_amplitude = np.sqrt(np.mean(E**2)) #np.sqrt(np.mean(E**2))
    Max_amplitude = np.max(E)
    print(Max_amplitude)
    
    # fig, ax = plt.subplots(2,1, figsize = (10,8), sharex=True, gridspec_kw = {'height_ratios': [2.5,1]})
    
    # ax[0].set_title("Variation from Mean Total Energies in FDM over Time", fontdict={'fontsize' : 15})
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
    ax.plot(k[1:]**(1/3), E[1:],"-")
    ax.set_xlabel("Cube Root Frequency $\\sqrt[3]{k}$",fontsize=12)
    ax.set_ylabel("Power", fontsize=12)
    
    plt.grid(True)
    plt.yscale("log")
    plt.show()

def v_distribution(z,L,chi,r,mu, plot=False):
    dz = z[1]-z[0]
    
    # eta = 10*r #resolution for Husimi
    k = 2*np.pi*np.fft.fftfreq(len(z),dz)
    #rescale wavenumber k to velocity v:
    hbar = 1
    v = k*(hbar/mu)
    dv = v[1]-v[0]
    # x_min, x_max = np.min(z), np.max(z)
    # v_min, v_max = np.min(v), np.max(v)
    # F = Wave.Husimi_phase(chi,z,dz,L,eta)

    # v_dist = []
    # for i in range(len(F)):
    #     row = F[i,:]
    #     amplitude = np.sum(row)*dz
    #     v_dist.append(amplitude)

    # v_dist = np.array(v_dist[::-1])
    rho = mu*np.abs(chi)**2
    v_dist = np.abs( np.fft.fft(rho) )**2

    #normalize:
    Norm_const = np.sum(dv*v_dist)
    v_dist = v_dist/Norm_const
    
    if plot==True:
        plt.plot(v,v_dist)
        plt.ylabel("Density (Normalized)")
        plt.xlabel("Velocity $v$")
        plt.title("FDM Virialized Velocity Distribution")
        plt.show()
        
    return v_dist

def rms_stuff(z,L,chi,v_dist,mu, plot=False):
    dz = z[1]-z[0]
    #Must re-center |chi|^2:
    #First, get center:
    weight = np.abs(chi)**2
    center = dz*np.sum([zz*w for zz,w in zip(z,weight)])
    center = center/np.sum(dz*np.abs(chi)**2)
    if plot==True:
        plt.plot(z,np.abs(chi)**2)
        plt.scatter(center,0,s = 10, c = "red")
        plt.show()
    
    # #Second, find index of center
    # index = 0
    # for zz in z:
    #     if zz < center:
    #         index+=1
    #     elif zz >= center:
    #         break
    # #Third, slice nad re-arrange chi:
    # if index <= len(z)/2: #then you have to take the right end and bring it to the left
    #     shift_i = int(np.floor(len(z)/2)) - index
    #     chi = np.append(chi[-shift_i:],chi[0:len(chi)-shift_i])    
    # else:
    #     shift_i = index - int(np.floor(len(z)/2))
    #     chi = np.append(chi[len(chi)-shift_i:],chi[0:len(chi)-shift_i])    

    # #chi = np.array()
    # plt.plot(z,chi)

    #normalize |chi|^2 to a probability density:
    p_z = np.abs(chi)**2
    Norm_const = np.sum(dz*p_z)
    p_z = p_z/Norm_const
    
    k = 2*np.pi*np.fft.fftfreq(len(z),dz)
    #rescale wavenumber k to velocity v:
    hbar = 1
    v = k*(hbar/mu)
    
    N1 = len(p_z)
    N2 = len(v_dist)

    #Calculated as standard deviation rather than rms directly
    z_rms = np.sqrt(
        np.sum(z**2 * p_z) / N1
    ) - center
    v_rms = np.sqrt(
        np.sum(v**2 * v_dist) / N2
    )

    return z_rms, v_rms
