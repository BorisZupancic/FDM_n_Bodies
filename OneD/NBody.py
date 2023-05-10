from imp import acquire_lock
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
import os
import cv2 
from PIL import Image 
import OneD.FDM as FDM
import OneD.Global as GF

from scipy.interpolate import interp1d

class stars:
    '''Class to represent an array of stars. Attributes:
    
    mass: Each "star" has the same mass  
    
    x: (numpy) array of positions

    v: (numpy) array of velocities
    '''

    def __init__(self,masses,x,v):
        self.mass = masses #masses of star(s)
        self.x = x # positions
        self.v = v # velocities
            
    def kick(self,g,dt):
        self.v += dt*g#KICK velocity
        
    def drift(self,dt):
        self.x += self.v*dt #DRIFT the position
    
    def reposition(self,L):
        div = (self.x // (L/2))
        remainder = self.x % (L/2)
        if div % 2 == 0: #check if modulo is even
            self.x = remainder 
        else: #if modulo is odd, further check:
            self.x = remainder-L/2
    
    def get_W(self,z,phi):
        sigma = self.mass
        dz = z[1]-z[0]
        N = len(phi)
        #print(np.min(z))
        #Find the potential energies of each "star" in stars
        #zz = self.x
        phi_interp = interp1d(z,phi)
        Potentials = phi_interp(self.x)

        # n = int((zz+L/2)//dz)
        # rem = (zz+L/2) % dz 
        # Potentials = np.zeros_like(self.x)
        # if n < N-1:
        #     Potential = phi[n] + rem*(phi[n+1]-phi[n])/dz
        # elif n == N-1:
        #     Potential = phi[-1] + rem*(phi[0]-phi[-1])/dz
        
        W = sigma*Potentials
        return W

    def get_K(self):
        K = 0.5* self.mass * self.v **2
        return K

def collectEnergies(K1,W1,K2=None,W2=None):
    
    return
        

##########
# Define an algorithm to calculate the acceleration at each point of the grid
#... given a set of stars
########

def grid_count(stars,L,z):
    '''Count the particles on the mesh-grid.'''
    N = len(z)
    dz = z[1]-z[0]
    bin_counts, bin_edges = np.histogram(stars.x, bins = N, range = (-L/2-dz/2,L/2 + dz/2), weights = stars.mass)
    return bin_counts

def particle_density(stars, L, z, variable_mass):
    '''Get the density of the particle distribution on the mesh-grid.'''
    if variable_mass[0] == True:
        stars1 = stars[0]
        stars2 = stars[1]
        # sigma1 = stars1.mass
        # sigma2 = stars2.mass

        grid_counts1 = grid_count(stars1,L,z)
        grid_counts2 = grid_count(stars2,L,z)

        dz = z[1]-z[0]

        rho_part1 = grid_counts1/dz 
        rho_part2 = grid_counts2/dz
        #rho_part = (grid_counts1*sigma1 + grid_counts2*sigma2)/dz
    else:
        # sigma = stars.mass 
        grid_counts = grid_count(stars,L,z)
        dz = z[1]-z[0]
        rho_part1 = (grid_counts/dz)
        rho_part2 = np.zeros_like(rho_part1)
    return rho_part1, rho_part2
    
def acceleration(phi,L,type):
    grad = GF.gradient(phi,L,type = type)#GF.fourier_gradient(phi,L)
    acceleration = -grad
    return acceleration  

def simulate_NBody_FixedPhi(stars,phi,g,z,L,dtau,tau_stop,Directory,folder_name):
    dtau = 0.1
    tau_stop = 20
    time = 0
    i = 0 #counter, for saving images
    while time <= tau_stop:
        fig,ax = plt.subplots(1,2)
        plt.suptitle(f"Time {round(dtau*i,1)}")
        
        ax[0].plot(z,phi,label = "Potential")
            
        for my_star in stars:
            #1. PLOT FIRST
            x = my_star.x
            v = my_star.v

            ax[0].plot(x,0,".")
            ax[0].set_xlim([-L/2,L/2])
            #ax[0].set_ylim([np.min(positions),np.max(positions)])
            ax[0].legend()

            ax[1].plot(x,v,".")
            ax[1].set_xlim([-L/2,L/2])
            ax[1].set_ylim([-0.5,0.5])
            #ax[1].legend()
            
            #2. THEN EVOLVE SYSTEM
            my_star.evolve_star_dynamics(g,dtau)

        #now save it as a .jpg file:
        folder = Directory + "/" + folder_name
        filename = 'ToyModelPlot' + str(i+1).zfill(4) + '.jpg';
        plt.savefig(folder + "/" + filename)  #save this figure (includes both subplots)
        
        plt.close() #close plot so it doesn't overlap with the next one
        time += dtau
        i += 1