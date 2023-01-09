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

Directory = os.getcwd()
print(Directory + " Contains:")
print(os.listdir())
print("Input Name of Folder within dirctory ' "+Directory+"' ")
folder_name = input()

Num_stars = 1
r=1
#WRITE TO VIDEO
video_name = folder_name + "_Video.mp4"
fps = 10
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
print("Now Saving Video")
GF.animate(fourcc,Directory,folder_name,video_name,fps)
print("Video Saved.")
