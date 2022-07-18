import time
st = time.time()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
import os
import subprocess
import cv2 
from PIL import Image 

#Import My Library
My_Package_PATH = "/home/boris/Documents/Research/FDM_n_Bodies"
import sys
sys.path.insert(1, My_Package_PATH)
import OneD.Wave as Wave
import OneD.NBody as NB
import OneD.Global as GF

#Set up Directory for saving files/images/videos
# Will not rename this again
dirExtension = "1D_Codes/Non-Dim/Programs"
Directory = os.getcwd()+"/"+dirExtension #os.curdir() #"/home/boris/Documents/Research/Coding/1D codes/Non-Dim"
print(Directory)

############################################
# SET UP: shared by Wave and N-Body scenarios
############################################
#Set scales and parameters:
L_s = 1 #kpc
v_s = 1 #100km/s
G = 1
hbar = 1

print(f"Length scale L = {L_s}")
print(f"Velocity scale v = {v_s}")

T = L_s/v_s
print(f"Time scale T = {T}")

M_s = L_s*v_s**2
print(f"Mass scale M = {M_s}")

#L, choice = GF.Startup_Choice()
L, mu, Num_bosons, r, sigma, Num_stars = GF.StartupV2(hbar, L_s, v_s)
m = mu*M_s
percent_FDM = Num_bosons * mu / (Num_bosons * mu + Num_stars * sigma)

#Set up Grid
L = L*L_s #new length. Length of the box
N = 10**3
z = np.linspace(-L/2,L/2,N)
dz = z[1]-z[0]

################
#PROMPT FOR FULL SIMULATION OR SNAPSHOTS
#Also prompt for fixed potential

print("")
print("Do you want a fixed potetial (phi = 0.5*sigma*(2z/L)**2 - 1)? Choose [y/n]")
fixed_phi = input()
if fixed_phi == 'Y' or fixed_phi == 'y' or fixed_phi == None:
    fixed_phi = True
if fixed_phi == 'n':
    fixed_phi = False
print("")
print("Do you want the full simulation [1] or snapshots [2]? Choose [1/2]")
sim_choice2 = int(input())
print("")

#For te
################

#SET UP FOLDERS:
if sim_choice2 == 1: 
    folder_name = f"FDM{percent_FDM}_r{r}_Images"
    if Num_bosons == 0:
        folder_name = f"{Num_stars}ParticlesOnly_Images"
    elif Num_stars == 0:
        folder_name = f"OnlyFDM_r{r}_Images"
elif sim_choice2 == 2:
    folder_name = f"FDM{percent_FDM}_r{r}_Snapshots"
    if Num_bosons == 0:
        folder_name = f"{Num_stars}ParticlesOnly_Snapshots"
    elif Num_stars == 0:
        folder_name = f"OnlyFDM_r{r}_Snapshots"

#print(os.path.exists(dirExtension+"/"+folder_name))
if os.path.exists(dirExtension+"/"+folder_name) == True:
    for file in os.listdir(Directory+"/"+folder_name):
        os.remove(Directory+"/"+folder_name+"/"+file)
    os.rmdir(Directory+"/"+folder_name)    
os.mkdir(Directory+"/"+folder_name)

#RUN SIMULATION/CALCULATION
print("Calculating and Plotting...")
#folder_name = "bla" 
#Whether to track stars or not:
track_stars = False
if Num_stars != 0:
    track_stars = True

stars, chi, K_storage, W_storage, K_5stars_storage, W_5stars_storage, centroids = GF.run_FDM_n_Bodies(sim_choice2, z,L,dz,mu, Num_bosons, r, sigma,Num_stars,v_s,L_s,Directory,folder_name, absolute_PLOT = True, track_stars = track_stars, track_centroid=True,fixed_phi = fixed_phi)
print("Calculation and Plotting Done. Now Saving Data...")

#os.chdir(Directory)#+"/"+dirExtension)

if Num_bosons == 0:
    np.savetxt(f"StarsOnly_Pos.csv",[star.x for star in stars], delimiter = ",")
    np.savetxt(f"StarsOnly_Vel.csv",[star.v for star in stars], delimiter = ",")
    np.savetxt(f"K_Energies.csv", K_storage, delimiter = ",")
    np.savetxt(f"W_Energies.csv", W_storage, delimiter = ",")
    np.savetxt(f"K_5stars_Energies.csv", K_5stars_storage, delimiter = ",")
    np.savetxt(f"W_5stars_Energies.csv", W_5stars_storage, delimiter = ",")
    np.savetxt(f"Chi.csv", chi,delimiter = ",")
    np.savetxt(f"Centroids.csv",centroids,delimiter = ',')
elif Num_stars == 0:
    #np.savetxt(f"Stars_Pos_m{m}.csv",[star.x for star in stars], delimiter = ",")
    #np.savetxt(f"Stars_Vel_m{m}.csv",[star.v for star in stars], delimiter = ",")
    np.savetxt(f"FuzzyOnlyChi_r{r}.csv", chi)
    #np.savetxt(f"Energies_m{m}.csv", E_storage, delimiter = ",")
elif Num_bosons!=0 and Num_stars!=0:
    np.savetxt(f"Stars_Pos.csv",[star.x for star in stars], delimiter = ",")
    np.savetxt(f"Stars_Vel.csv",[star.v for star in stars], delimiter = ",")
    np.savetxt(f"Chi.csv", chi)
    np.savetxt(f"K_Energies.csv", K_storage, delimiter = ",")
    np.savetxt(f"W_Energies.csv", W_storage, delimiter = ",")
    np.savetxt(f"K_5stars_Energies.csv", K_5stars_storage, delimiter = ",")
    np.savetxt(f"W_5stars_Energies.csv", W_5stars_storage, delimiter = ",")
    np.savetxt(f"Centroids.csv",centroids,delimiter = ',')
print("Data Saved.")

et = time.time()
elapsed_time = et-st
print(f"Executed in {elapsed_time} seconds = {elapsed_time/60} minutes.")
properties = [["Time Elapsed:", elapsed_time],
              ["Box Length:", L],
              ["Boson Mass:",mu],
              ["Number of bosons:",Num_bosons],
              ["Particle mass:",sigma],
              ["Number of Particles:",Num_stars]]
np.savetxt(f"Properties.csv", properties, delimiter = ",", fmt = "%s")

if sim_choice2 == 1:
    print("Now Saving Video")
    #WRITE TO VIDEO
    video_name = "Video"
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    GF.animate(fourcc,Directory,folder_name,video_name,fps)
    print("Video Saved.")
# if sim_choice2 == 1:
#     subprocess.call(["xdg-open", "FDM_n_Body.mp4"])
# elif sim_choice2 == 2:
#     subprocess.call(["xdg-open", "FDM_n_Body_Snapshots.mp4"]) 

