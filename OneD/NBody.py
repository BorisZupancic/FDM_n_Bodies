from imp import acquire_lock
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
import os
import cv2 
from PIL import Image 
import OneD.WaveNonDim as ND
import OneD.GlobalFuncs as GF

class star:
    def __init__(self,id,mass,x,v):
        self.id = id #identification number
        self.mass = mass #mass of star
        self.x = x #it's "instantaneous" position
        self.v = v #it's "instantaneous" velocity
        #self.bin = bin #bin it resides in: index value between 0 and N-1

    def kick_star(self,g,dt):
        a = g(self.x)
        self.v += dt*a #KICK velocity

    def drift_star(self,dt):
        self.x += self.v*dt #DRIFT the position

    def evolve_star_dynamics(self,g,dt): #g: acceleration, dt: time-step
        a = g(self.x)
        self.v += 0.5*dt*a # KICK evolve velocity forward by half time-step
        
        self.x += self.v*dt #DRIFT the position

        a = g(self.x)
        self.v += 0.5*dt*a #KICK to get full new velocity(this will change for self-gravitating system)

# class star_collection:
#     def __init__(self, stars: star):
#         self.stars = stars
    
#     def evolve_dynamics(self,g,dt):
        

##########
# Define an algorithm to calculate the acceleration at each point of the grid
#... given a set of stars
########
def interp_bins(x,bin_counts):
    grid_counts = np.zeros_like(x)
    for i in range(len(x)):
        if i == 0 or i == len(x)-1:
            grid_counts[i] += 0.5*bin_counts[0] 
            grid_counts[i] += 0.5*bin_counts[-1]
        else: 
            grid_counts[i] += 0.5*bin_counts[i-1] #half goes to the end right end of the grid
            grid_counts[i] += 0.5*bin_counts[i]
    return grid_counts

def grid_count(stars,L,x):
    #Count the particles on the Mesh
    N = len(x)
    bin_counts, bin_edges = np.histogram([star.x for star in stars], bins = N-1, range = (-L/2,L/2))
    #interpolate bin_counts to counts on edges:
    grid_counts = interp_bins(x,bin_counts)
    return grid_counts

def acceleration(phi,L):
    grad = GF.fourier_gradient(phi,L)
    #grad = np.gradient(phi,dz)
    Force = -grad
    
    acceleration = Force#/m #acceleration per particle, at each point of the grid
    return acceleration#/L #divide by L to keep it non-dimensional

def g(star,acceleration,dz):
    a_grid = acceleration
    
    i = int(star.x//dz)
    rem = star.x % dz 
    value = a_grid[i] + rem*(a_grid[i]-a_grid[i-1])/dz
    return value

def accel_funct(a_grid,L,dz):
    def g(z):
        N = len(a_grid)
        j = int((z+L/2)//dz)
        rem = (z+L/2) % dz 
        value = 0
        if j < N-1:
            value = a_grid[j] + rem*(a_grid[j+1]-a_grid[j])/dz
        elif j == N-1:
            value = a_grid[-1]+rem*(a_grid[0]-a_grid[-1])/dz
        return value
    return g

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