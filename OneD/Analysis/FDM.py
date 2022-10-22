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


def plot_Energies(indices,Ks,Ws):
    #set the scale:
    Dy_1 = np.max(Ks)-np.min(Ks)
    Dy_2 = np.max(Ws)-np.min(Ws)
    Dy = np.max([Dy_1,Dy_2])
    
    fig, ax = plt.subplots(1,4, figsize = (15,5))
    
    y_midW = np.min(Ws)+Dy_2/2
    y_minW = y_midW - Dy/2
    y_maxW = y_midW + Dy/2
    ax[0].plot(indices, Ws,"o--")
    ax[0].set_title("FDM Potential")
    ax[0].set_ylim(y_minW,y_maxW)
    
    y_midK = np.min(Ks)+Dy_1/2
    y_minK = y_midK - Dy/2
    y_maxK = y_midK + Dy/2
    ax[1].plot(indices,Ks,"o--")
    ax[1].set_title("FDM Kinetic")
    ax[1].set_ylim(0,np.max(Ks))

    
    y_avg = np.mean(Ks+Ws)
    y_min = y_avg - Dy/2
    y_max = y_avg + Dy/2
    ax[2].plot(indices, Ks+Ws,"o--")
    ax[2].set_title("FDM Total Energy")
    ax[2].set_ylim(y_min,y_max)

    ax[3].plot(indices, np.abs(Ks/Ws),"o--")
    ax[3].set_title("$\\frac{K}{|W|}$")
    plt.show()

def v_distribution(z,L,chi,r,mu):
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
   
    print(np.sum(dv*v_dist))
    plt.plot(v,v_dist)
    plt.ylabel("Density (Normalized)")
    plt.xlabel("Velocity $v$")
    plt.title("FDM Virialized Velocity Distribution")
    plt.show()
    
    return v_dist

def rms_stuff(z,L,chi,v_dist,mu):
    dz = z[1]-z[0]
    #Must re-center |chi|^2:
    #First, get center:
    weight = np.abs(chi)**2
    center = dz*np.sum([zz*w for zz,w in zip(z,weight)])
    center = center/np.sum(dz*np.abs(chi)**2)
    print(center)
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
