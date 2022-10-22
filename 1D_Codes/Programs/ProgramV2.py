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
import OneD.FDM as FDM
import OneD.NBody as NB
import OneD.Global as GF

#Set up Directory for saving files/images/videos
# Will not rename this again
print("Specify Path to create folder for saving images.")
parent = str(input())

dirExtension = "1D_Codes/Programs"
Directory = os.getcwd()+"/"+dirExtension+"/"+parent #os.curdir() #"/home/boris/Documents/Research/Coding/1D codes/Non-Dim"
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
L, mu, Num_bosons, r, lambda_deB, R, sigma, Num_stars = GF.Startup(hbar, L_s, v_s)
m = mu*M_s
percent_FDM = Num_bosons * mu / (Num_bosons * mu + Num_stars * sigma)

print(f"lambda_deB = {lambda_deB}")

#Set up Grid
L = L*L_s #new length. Length of the box
N = 10**3 #number of grid points
if Num_bosons != 0: #want to determine the proper number of grid points to have
    lambda_deB = lambda_deB*R
    N_new = int(np.ceil(L/lambda_deB)+1)
    if N_new >= N:
        N = 12 *N_new #overwrite number of grid points
    #     N = 8*N_new #overwrite number of grid points
print(f"Number of Grid points: {N}")
z = np.linspace(-L/2,L/2,N)
dz = z[1]-z[0]

############################################################
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
print("Isolated [1] or Periodic [2] boundary conditions?")
bc_choice = int(input())
# if bc_choice == 1:
#     pass
# elif bc_choice == 2:
#     G = None
print("")
print("Do you want the full simulation [1] or snapshots [2]? Choose [1/2]")
sim_choice2 = int(input())
print("")


####################################################################
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

#print(os.path.exists(Directory+"/"+folder_name))
if os.path.exists(Directory+"/"+folder_name) == True:
    for file in os.listdir(Directory+"/"+folder_name):
        os.remove(Directory+"/"+folder_name+"/"+file)
    os.rmdir(Directory+"/"+folder_name)    
os.mkdir(Directory+"/"+folder_name)

#####################################################################
# Set-Up is Done. Simulation next.
#####################################################################
#RUN SIMULATION/CALCULATION
print("Calculating and Plotting...")

#Whether to track stars or not:
track_stars = False
track_stars_rms = False
if Num_stars != 0:
    track_stars = True
    track_stars_rms = True

#Create Initial Conditions:
print("Initial Conditions: Gaussian, Sine^2, or Spitzer? Enter [1,2,or 3]:")
ICs = float(input())
if ICs == 1:
    stars,chi = GF.gaussianICs(z, L, Num_bosons, sigma, Num_stars, v_s, L_s)
elif ICs == 2:
    stars,chi = GF.sine2_ICs(z, L, Num_bosons, sigma, Num_stars, v_s, L_s)
elif ICs == 3:
    xIC,vIC = GF.SpitzerICs(Num_stars)
    stars = [NB.star(i,sigma,xIC[i],vIC[i]) for i in range(Num_stars)]
    chi = np.zeros_like(z)

#Run simulation on Initial Conditions:
stars, chi, z_rms_storage, v_rms_storage, K_star_storage, W_star_storage, K_star_fine_storage, W_star_fine_storage, K_5stars_storage, W_5stars_storage, centroids, K_FDM_storage, W_FDM_storage= GF.run_FDM_n_Bodies(sim_choice2, bc_choice, z,L,dz,
                                                                                                      mu, Num_bosons, r, chi, 
                                                                                                      sigma,stars,
                                                                                                      v_s,L_s,
                                                                                                      Directory,folder_name, 
                                                                                                      absolute_PLOT = True, track_stars = track_stars, track_stars_rms = track_stars_rms, track_centroid=True, fixed_phi = fixed_phi, track_FDM=True)
print("Calculation and Plotting Done. Now Saving Data...")

############################
#Saving the Data
#os.chdir(Directory)#+"/"+dirExtension)

if Num_bosons == 0:
    np.savetxt(f"StarsOnly_Pos.csv",[star.x for star in stars], delimiter = ",")
    np.savetxt(f"StarsOnly_Vel.csv",[star.v for star in stars], delimiter = ",")
    np.savetxt(f"z_rms_storage.csv", z_rms_storage, delimiter = ",")
    np.savetxt(f"v_rms_storage.csv", v_rms_storage, delimiter = ",")
    np.savetxt(f"K_star_Energies.csv", K_star_storage, delimiter = ",")
    np.savetxt(f"W_star_Energies.csv", W_star_storage, delimiter = ",")
    np.savetxt(f"K_star_fine_Energies.csv", K_star_fine_storage, delimiter = ",")
    np.savetxt(f"W_star_fine_Energies.csv", W_star_fine_storage, delimiter = ",")
    
    if Num_stars>=5:
        np.savetxt(f"K_5stars_Energies.csv", K_5stars_storage, delimiter = ",")
        np.savetxt(f"W_5stars_Energies.csv", W_5stars_storage, delimiter = ",")
    np.savetxt(f"Chi.csv", chi,delimiter = ",")
    np.savetxt(f"Centroids.csv",centroids,delimiter = ',')
elif Num_stars == 0:
    np.savetxt(f"Chi.csv", chi, delimiter =",")
    np.savetxt(f"W_FDM_storage.csv", W_FDM_storage, delimiter =",")
    np.savetxt(f"K_FDM_storage.csv", K_FDM_storage, delimiter =",")
elif Num_bosons!=0 and Num_stars!=0:
    np.savetxt(f"Stars_Pos.csv",[star.x for star in stars], delimiter = ",")
    np.savetxt(f"Stars_Vel.csv",[star.v for star in stars], delimiter = ",")
    np.savetxt(f"z_rms_storage.csv", z_rms_storage, delimiter = ",")
    np.savetxt(f"v_rms_storage.csv", v_rms_storage, delimiter = ",")
    np.savetxt(f"Chi.csv", chi)
    np.savetxt(f"W_FDM_storage.csv", W_FDM_storage, delimiter =",")
    np.savetxt(f"K_FDM_storage.csv", K_FDM_storage, delimiter =",")
    np.savetxt(f"K_star_Energies.csv", K_star_storage, delimiter = ",")
    np.savetxt(f"W_star_Energies.csv", W_star_storage, delimiter = ",")
    np.savetxt(f"K_star_fine_Energies.csv", K_star_fine_storage, delimiter = ",")
    np.savetxt(f"W_star_fine_Energies.csv", W_star_fine_storage, delimiter = ",")
    
    if Num_stars>=5:
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
              ["Number of Particles:",Num_stars],
              ["FDM Fuzziness:",r],
              ["Grid Points:", N]]
np.savetxt(f"Properties.csv", properties, delimiter = ",", fmt = "%s")

if sim_choice2 == 1:
    print("Now Saving Video")
    #WRITE TO VIDEO
    video_name = f"r{r}_NumP_{Num_stars}_Video.mp4"
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    GF.animate(fourcc,Directory,folder_name,video_name,fps)
    print("Video Saved.")
# if sim_choice2 == 1:
#     subprocess.call(["xdg-open", "FDM_n_Body.mp4"])
# elif sim_choice2 == 2:
#     subprocess.call(["xdg-open", "FDM_n_Body_Snapshots.mp4"]) 

