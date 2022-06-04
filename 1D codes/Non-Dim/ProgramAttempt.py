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
import OneD.Waves.NonDim as ND
import OneD.NBody.NBody as NB
import OneD.Global.GlobalFuncs as GF

#Set up Directory for saving files/images/videos
# Will not rename this again
from pathlib import Path
Directory = os.getcwd()+"/1D codes/Non-Dim" #os.curdir() #"/home/boris/Documents/Research/Coding/1D codes/Non-Dim"
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

L, choice = GF.Startup_Choice()

#Set up Grid
L = L*L_s #new length. Length of the box
N = 10**3
z = np.linspace(-L/2,L/2,N)
dz = z[1]-z[0]

if choice == 1:
    mu, Num_bosons, r  = GF.Startup_Initial_Parameters(choice, hbar, L_s,v_s,M_s)
    
    #Calculate and Plot Everything:
    print("Calculating and Plotting...")
    folder_name = "SelfGrav_Images"
    GF.run_FDM(z,L,dz,mu,Num_bosons,r,v_s,L_s,Directory,folder_name)
    print("Calculation and Plotting Done. Now Saving Video...")

    #Animate the Plots
    folder_name = "SelfGrav_Images"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    m = mu*M_s
    video_name = f"SelfGrav_Mass{m}.mp4"
    fps = 10 #1/dtau
    GF.animate(fourcc,Directory,folder_name,video_name,fps) 
    print("Video Saved.")

    #Open the video
    subprocess.call(["xdg-open", "SelfGrav_Mass1.0.mp4"])

elif choice == 2:
    sigma, Num_stars = GF.Startup_Initial_Parameters(choice, hbar, L_s,v_s, M_s)
    print("Calculating and Plotting...")
    tau_collapse = 0.044
    GF.run_NBody(z,L,dz,sigma,Num_stars,v_s,L_s,Directory)

    print("Calculation and Plotting Done. Now Saving Video...")
    folder_name = "SelfGrav_NBody_Images"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = "SelfGrav_NBody.mp4"
    fps = 10 #1/dtau
    GF.animate(fourcc,Directory,folder_name,video_name,fps)
    print("Video Saved.")
    subprocess.call(["xdg-open", "SelfGrav_NBody.mp4"])

