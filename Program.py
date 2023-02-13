import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
import os
import subprocess
import cv2 
from PIL import Image 

#Import My Library
My_Package_PATH = "/home/boris/Documents/Research/FDM_Codes/FDM_n_Bodies"
import sys
sys.path.insert(1, My_Package_PATH)
import OneD.FDM as FDM
import OneD.NBody as NB
import OneD.Global as GF
import OneD.Init_C as IC

#Set up Directory for saving files/images/videos
# Will not rename this again
print("Specify Path to create folder for saving images.")
parent = str(input())

# dirExtension = "1D_Codes/Programs"
# Directory = os.getcwd()+"/"+dirExtension+"/"+parent #os.curdir() #"/home/boris/Documents/Research/Coding/1D codes/Non-Dim"
Directory = os.getcwd()+"/"+parent #os.curdir() #"/home/boris/
print(Directory)

############################################
# SET UP:
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

fixed_phi, bc_choice, sim_choice2, dynamical_times, ICs = IC.startup()

print("---Creating Initial Conditions---")

z, stars, chi, mu, Num_bosons, r, T_Dynamical, zmax, vmax = IC.init(hbar,L_s, v_s, ICs)    
L = z[-1]-z[0]
dz = z[1]-z[0]
Num_stars = len(stars.x)
mass_FDM = mu*dz*np.sum(np.abs(chi)**2)
mass_part = stars.mass*Num_stars
Total_mass = mass_part + mass_FDM
print(f"Total_mass = {Total_mass}")
percent_FDM = mass_FDM/Total_mass
print("")

print("Variable Star Masses [y/n]? (Split in two)")
variable_mass = input()


if variable_mass == 'y':
    print("[Y]")
    print("Input fraction of stars to convert to heavy:")
    fraction = input()
    print(fraction)
    print("Splitting Stars in Two...")
    num, denom = fraction.split('/')
    fraction = float(num)/float(denom)
    num_to_change = int(np.floor(fraction*Num_stars))
    print(f"Number of heavier particles: {num_to_change}")
    sigma1 = (Total_mass/2)/num_to_change #sigma*2
    sigma2 = (Total_mass/2)/(Num_stars-num_to_change) #sigma*fraction

    from numpy.random import default_rng
    rng = default_rng()
    i_s = rng.choice(Num_stars, size=num_to_change, replace=False)
    print(len(i_s))
    part1 = np.array([[stars.x[i],stars.v[i]] for i in i_s])
    part2 = np.array([[stars.x[i],stars.v[i]] for i in range(Num_stars) if i not in i_s])
    # for i in range(Num_stars):
    #     if i in i_s:
    #         part1 = np.append(part1,[stars.x[i],stars.v[i]],axis = 1)
    #     else:
    #         part2 = np.append(part2,[stars.x[i],stars.v[i]],axis = 1)
    stars1 = NB.stars(sigma1,part1[:,0],part1[:,1])
    stars2 = NB.stars(sigma2,part2[:,0],part2[:,1])

    #re-center position and velocity centroids:
    z1_centroid, z2_centroid = [np.mean(stars1.x),np.mean(stars2.x)]
    v1_centroid, v2_centroid = [np.mean(stars1.v),np.mean(stars2.v)]
    stars1.x += -z1_centroid
    stars1.v += -v1_centroid
    stars2.x += -z2_centroid
    stars2.v += -v2_centroid
    
    stars = [stars1, stars2]
    print(f"len(stars1.x) = {len(stars[0].x)}")
    print(f"len(stars2.x) = {len(stars[1].x)}")
    print(f"stars1.mass = {stars[0].mass}")
    print(f"sigma1 = {sigma1}")
    
    variable_mass = [True,fraction,sigma1,sigma2]
else:
    variable_mass = [False]
    print("[n]")

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
        if variable_mass[0] == False:
            folder_name = f"Sigma{stars2}_{Num_stars}ParticlesOnly_Snapshots"
        else:
            folder_name = f"SigmaRatio({sigma1/sigma2})_{Num_stars}ParticlesOnly_Snapshots"
    elif Num_stars == 0:
        folder_name = f"OnlyFDM_r{r}_Snapshots"

print(os.getcwd())
print(os.path.exists(Directory+"/"+folder_name))
if os.path.exists(Directory+"/"+folder_name) == True:
    for file in os.listdir(Directory+"/"+folder_name):
        os.remove(Directory+"/"+folder_name+"/"+file)
    os.rmdir(Directory+"/"+folder_name)    
print(Directory+"/"+folder_name)
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


st = time.process_time() #Start Tracking CPU Time
#Run simulation on Initial Conditions:

snapshot_indices, stars, chi, z_rms_storage, v_rms_storage, K_star_storage, W_star_storage, W_2_star_storage, K_star_fine_storage, W_star_fine_storage, part_centroids, fdm_centroids, K_FDM_storage, W_FDM_storage= GF.run_FDM_n_Bodies(sim_choice2, dynamical_times, T_Dynamical, bc_choice, z,
                                                                                                      mu, Num_bosons, r, chi, 
                                                                                                      stars,
                                                                                                      v_s,L_s, zmax, vmax,
                                                                                                      Directory,folder_name, 
                                                                                                      absolute_PLOT = True, track_stars = track_stars, track_stars_rms = track_stars_rms, track_centroid=True, fixed_phi = fixed_phi, track_FDM=True, variable_mass = variable_mass)
print("Calculation and Plotting Done.")
et = time.process_time()
elapsed_time = et-st
print(f"Executed in {elapsed_time} seconds = {elapsed_time/60} minutes = {elapsed_time/3600} hours.")
print("Now Saving Data...")
############################
#Saving the Data
#os.chdir(Directory)#+"/"+dirExtension)

if Num_bosons == 0:
    if variable_mass[0]==True:
        np.savetxt(f"StarsOnly_Pos.csv",[*stars[0].x,*stars[1].x], delimiter = ",")
        np.savetxt(f"StarsOnly_Vel.csv",[*stars[0].v,*stars[1].v], delimiter = ",")
    else:
        np.savetxt(f"StarsOnly_Pos.csv",stars.x, delimiter = ",")
        np.savetxt(f"StarsOnly_Vel.csv",stars.v, delimiter = ",")
    np.savetxt(f"z_rms_storage.csv", z_rms_storage, delimiter = ",")
    np.savetxt(f"v_rms_storage.csv", v_rms_storage, delimiter = ",")
    np.savetxt(f"K_star_Energies.csv", K_star_storage, delimiter = ",")
    np.savetxt(f"W_star_Energies.csv", W_star_storage, delimiter = ",")
    np.savetxt(f"W_2_star_Energies.csv", W_star_storage, delimiter = ",")

    np.savetxt(f"K_star_fine_Energies.csv", K_star_fine_storage, delimiter = ",")
    np.savetxt(f"W_star_fine_Energies.csv", W_star_fine_storage, delimiter = ",")
    
    np.savetxt(f"Chi.csv", chi,delimiter = ",")
    np.savetxt(f"Particle_Centroids.csv",part_centroids,delimiter = ',')
elif Num_stars == 0:
    np.savetxt(f"Chi.csv", chi, delimiter =",")
    np.savetxt(f"W_FDM_storage.csv", W_FDM_storage, delimiter =",")
    np.savetxt(f"K_FDM_storage.csv", K_FDM_storage, delimiter =",")
    np.savetxt(f"FDM_Centroids.csv",fdm_centroids,delimiter = ',')
elif Num_bosons!=0 and Num_stars!=0:
    if variable_mass[0]==True:
        np.savetxt(f"StarsOnly_Pos.csv",[*stars[0].x,*stars[1].x], delimiter = ",")
        np.savetxt(f"StarsOnly_Vel.csv",[*stars[0].v,*stars[1].v], delimiter = ",")
    else:
        np.savetxt(f"StarsOnly_Pos.csv",stars.x, delimiter = ",")
        np.savetxt(f"StarsOnly_Vel.csv",stars.v, delimiter = ",")
    np.savetxt(f"z_rms_storage.csv", z_rms_storage, delimiter = ",")
    np.savetxt(f"v_rms_storage.csv", v_rms_storage, delimiter = ",")
    np.savetxt(f"Chi.csv", chi)
    np.savetxt(f"FDM_Centroids.csv",fdm_centroids,delimiter = ',')
    np.savetxt(f"W_FDM_storage.csv", W_FDM_storage, delimiter =",")
    np.savetxt(f"K_FDM_storage.csv", K_FDM_storage, delimiter =",")
    np.savetxt(f"K_star_Energies.csv", K_star_storage, delimiter = ",")
    np.savetxt(f"W_star_Energies.csv", W_star_storage, delimiter = ",")
    np.savetxt(f"W_2_star_Energies.csv", W_star_storage, delimiter = ",")

    np.savetxt(f"K_star_fine_Energies.csv", K_star_fine_storage, delimiter = ",")
    np.savetxt(f"W_star_fine_Energies.csv", W_star_fine_storage, delimiter = ",")
    
    np.savetxt(f"Particle_Centroids.csv",part_centroids,delimiter = ',')


if variable_mass[0] == True:
    properties = [["Time Elapsed:", elapsed_time],
              ["Box Length:", L],
              ["Boson Mass:",mu],
              ["Number of bosons:",Num_bosons],
              ["Particle mass(es):",stars[0].mass],
              ["Variable mass:", variable_mass[0]],
              ["Variable mass fraction", variable_mass[1]],
              ["sigma1", variable_mass[2]],
              ["sigma2", variable_mass[3]],
              ["Number of Particles:",Num_stars],
              ["FDM Fuzziness:",r],
              ["Grid Points:", len(z)],
              ["Snapshot Indices:", snapshot_indices]]
else:
    properties = [["Time Elapsed:", elapsed_time],
              ["Box Length:", L],
              ["Boson Mass:",mu],
              ["Number of bosons:",Num_bosons],
              ["Particle mass(es):",stars.mass],
              ["Variable mass:", variable_mass[0]],
              ["Variable mass fraction", 0],
              ["sigma1", stars.mass],
              ["sigma2", 0],
              ["Number of Particles:",Num_stars],
              ["FDM Fuzziness:",r],
              ["Grid Points:", len(z)],
              ["Snapshot Indices:", snapshot_indices]]
np.savetxt(f"Properties.csv", properties, delimiter = ",", fmt = "%s")

print("Data Saved.")

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

