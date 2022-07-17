import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
import os
import subprocess
import cv2 
from PIL import Image 

#Import My Library
My_Package_PATH = "/home/boris/Documents/Research/Coding"
import sys
sys.path.insert(1, My_Package_PATH)
import OneD.WaveNonDim as ND
import OneD.NBody as NB
import OneD.GlobalFuncs as GF

#Set up Directory for saving files/images/videos
# Will not rename this again
dirExtension = "1D_Codes/Non-Dim/Program"
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
L, mu, Num_bosons, r, sigma, Num_stars = GF.Startup(hbar, L_s, v_s, M_s)
m = mu*M_s

#Set up Grid
L = L*L_s #new length. Length of the box
N = 10**3
z = np.linspace(-L/2,L/2,N)
dz = z[1]-z[0]

################
#PROMPT FOR FULL SIMULATION OR SNAPSHOTS
print("")
print("Do you want the full simulation [1] or snapshots [2]? Choose [1/2]")
sim_choice2 = int(input())
print("")
################

#SET UP FOLDERS:
if sim_choice2 == 1: 
    folder_name = f"FuzzyMass{m}_Images"
    if Num_bosons == 0:
        folder_name = "ParticlesOnly_Images"
    elif Num_stars == 0:
        folder_name = f"OnlyFuzzyMass{m}_Images"
elif sim_choice2 == 2:
    folder_name = f"FuzzyMass{m}_Snapshots"
    if Num_bosons == 0:
        folder_name = "ParticlesOnly_Snapshots"
    elif Num_stars == 0:
        folder_name = f"OnlyFuzzyMass{m}_Snapshots"

#print(os.path.exists(dirExtension+"/"+folder_name))
if os.path.exists(dirExtension+"/"+folder_name) == True:
    for file in os.listdir(Directory+"/"+folder_name):
        os.remove(Directory+"/"+folder_name+"/"+file)
    os.rmdir(Directory+"/"+folder_name)    
os.mkdir(Directory+"/"+folder_name)

#RUN SIMULATION/CALCULATION
print("Calculating and Plotting...")
absolute_PLOT = True
stars, chi = GF.run_FDM_n_Bodies(sim_choice2, z,L,dz,mu, Num_bosons, r, sigma,Num_stars,v_s,L_s,Directory,folder_name)
print("Calculation and Plotting Done. Now Saving Video...")

#SET UP VIDEO NAMES
if sim_choice2 == 1:
    video_name = f"FuzzyMass{m}_Video.mp4"
    fps = 10 #1/dtau
    if Num_bosons == 0:
        video_name = "Particles_Video.mp4"
    elif Num_stars == 0:
        video_name = f"OnlyFuzzyMass{m}_Video.mp4"
elif sim_choice2 == 2:
    video_name = f"FuzzyMass{m}_Snapshots.mp4"
    fps = 1
    if Num_bosons == 0:
        video_name = "Particles_Snapshots.mp4"
    elif Num_stars == 0:
        video_name = f"OnlyFuzzyMass{m}_Snapshots.mp4"

#WRITE TO VIDEO
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
GF.animate(fourcc,Directory,folder_name,video_name,fps)
print("Video Saved.")
# if sim_choice2 == 1:
#     subprocess.call(["xdg-open", "FDM_n_Body.mp4"])
# elif sim_choice2 == 2:
#     subprocess.call(["xdg-open", "FDM_n_Body_Snapshots.mp4"])
