import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
import os
import cv2 
from PIL import Image 
    
def fourier_gradient(rho,mass,length):
    n = len(rho)
    m = mass
    L = length 

    #1. FFT the density (perturbation)
    rho_avg = np.mean(rho)
    P = rho-rho_avg
    P_n = np.fft.rfft(P,n) #fft for real input
    
    #2. Compute the fourier coefficients of phi
    grad_n = np.array([]) #empty for storage. Will hold the fourier coefficients
    for nn in range(len(P_n)):
        if nn == 0:
            val = 0 #the "Jean's Swindle"
        if nn >=1: #for the positive frequencies
            k = 2*np.pi*nn/L
            val = -P_n[nn]/k
        grad = np.append(grad_n,val)
    
    #3. IFFT back to get Potential
    grad = np.fft.irfft(grad_n,n) #use Phi_n as Fourier Coefficients

    return grad

#Determine the potential from poisson's equation using fourier method
def real_fourier_potential(psi,mass,length):
    n = len(psi)
    m = mass 
    L = length #length of box

    #1. FFT the norm-squared of the wave-function (minus it's mean background)
    rho = m*np.absolute(psi)**2 #this is the density
    rho_avg = np.mean(rho)
    P = rho-rho_avg
    P_n = np.fft.rfft(P,n) #fft for real input
    
    #2. Compute the fourier coefficients of phi
    Phi_n = np.array([]) #empty for storage. Will hold the fourier coefficients
    for nn in range(len(P_n)):
        if nn == 0:
            val = 0 #the "Jean's Swindle"
        if nn >=1: #for the positive frequencies
            k = 2*np.pi*nn/L
            val = -P_n[nn]/k**2
        Phi_n = np.append(Phi_n,val)
    
    #3. IFFT back to get Potential
    Phi = np.fft.irfft(Phi_n,n) #use Phi_n as Fourier Coefficients
    return Phi


def kick(psi,phi,dt,hbar,m):
    f = -1j*(m/hbar)*(dt/2)
    u = np.exp(f*phi)
    psi_new = np.multiply(u,psi)
    return psi_new

def drift(psi,dx,dt,hbar,m):
    n = len(psi)

    psi_n= np.fft.fft(psi,n)
    k = 2*np.pi*np.fft.fftfreq(n,dx) #wave-number in fourier domain

    f = -1j*(hbar/m)*(dt/2)
    u = np.exp(f*k**2)
    psi_new_n = np.multiply(u,psi_n)

    psi_new = np.fft.ifft(psi_new_n,n)
    return psi_new

def time_evolve(psi,phi,dx,dt,hbar,m,length):
    L = length 
    #1. Kick in current potential
    psi_new = kick(psi,phi,dt,hbar,m)

    #2. Drift in differential operator
    psi_new = drift(psi_new,dx,dt,hbar,m)

    #3. Update potential
    phi_new = real_fourier_potential(psi_new,m,L)

    #4. Kick in updated potential
    psi_new = kick(psi_new,phi_new,dt,hbar,m)

    #5. Calculate updated density perturbation
    rho_new = m*np.absolute(psi_new)**2
    P_new = rho_new-np.mean(rho_new)

    return psi_new,phi_new,P_new

def time_evolveV2_NonDim(psi,phi,dX,dT,m):
    #1. Kick in current potential
    psi_new = kick_NonDim(psi,phi,dT/2)

    #2. Drift in differential operator
    psi_new = drift_NonDim(psi_new,dX,dT)

    #3. Update potential
    phi_new = real_fourier_potential(psi_new,m)

    #4. Kick in updated potential
    psi_new = kick_NonDim(psi_new,phi_new,dT/2)

    #5. Calculate updated density perturbations
    rho_new = m*np.absolute(psi_new)**2
    P_new = rho_new-np.mean(rho_new)

    #6. Update potential again
    phi_new = real_fourier_potential(psi_new,m)
    
    return psi_new,phi_new,P_new

def time_evolve_FixedPhi(psi,phi,dx,dt,hbar,m):
    #1. Kick in current potential
    psi_new = kick(psi,phi,dt,hbar,m)

    #2. Drift in differential operator
    psi_new = drift(psi_new,dx,dt,hbar,m)

    #3. Kick in potential, again
    psi_new = kick(psi_new,phi,dt,hbar,m)

    #4. Calculate updated density perturbation
    rho_new = m*np.absolute(psi_new)**2
    P_new = rho_new-np.mean(rho_new)

    return psi_new,phi,P_new

## FOR NON DIMENSIONIZED, FIXED PHI SYSTEMS ONLY

def kick_NonDim(psi,phi,dT):
    f = -1j*dT
    u = np.exp(f*phi)
    psi_new = np.multiply(u,psi)
    return psi_new

def drift_NonDim(psi,dX,dT):
    n = len(psi)

    psi_n= np.fft.fft(psi,n)
    k = 2*np.pi*np.fft.fftfreq(n,dX) #wave-number in fourier domain

    f = -1j*dT
    u = np.exp(f*k**2)
    psi_new_n = np.multiply(u,psi_n)

    psi_new = np.fft.ifft(psi_new_n,n)
    return psi_new


def time_evolve_FixedPhi_NonDim(psi,phi,dX,dT,m):
    #1. Kick in potential
    psi_new = kick_NonDim(psi,phi,dT/2)

    #2. Drift in differential operator
    psi_new = drift_NonDim(psi_new,dX,dT)

    psi_new = kick_NonDim(psi_new,phi,dT/2)
    #3. Calculate updated density perturbation
    rho_new = m*np.absolute(psi_new)**2
    P_new = rho_new-np.mean(rho_new)

    return psi_new,phi,P_new

###################################################################################
# FOR PHASE-SPACE DISTRIBUTION STUFF
###################################################################################
def Husimi_phase(psi,x,dx,hbar):
    N = len(psi)
    eta = 1/(2*np.pi*N)**0.5
    A = 1/(2*np.pi*hbar)**0.5 
    B = 1/(np.pi*eta**2)**0.25 

    k = np.fft.fftfreq(len(x),dx)
    
    f_s = np.ndarray((N,N), dtype = complex)
    for i in range(len(x)):
        x_0 = x[i]

        g = np.exp(-(x_0-x)**2) / (2*eta**2)
        f = np.fft.ifft(np.multiply(psi,g))
        f = A*B*f #np.multiply(np.exp(1j*k*x_0/2),f)

        f = np.append(f[N//2:N],f[0:N//2])
        
        f_s[i] = f

    F_s = np.absolute(f_s)**2 
    F_s = np.transpose(F_s)
    
    return F_s


def phase_distribution(psi,x):
    N = len(psi)
    std = 0.5*np.sqrt(N)
    #do convuolution at each x_0 in x:
    f_s = np.ndarray((N,N), dtype = complex)
    for i in range(len(x)):
        x_0 = x[i]

        g = np.sqrt(2/(N*np.pi))*np.exp(-2*(x-x_0)**2 /N)
        f = np.fft.fft(np.multiply(psi,g))
        f = np.append(f[N//2:N],f[0:N//2])
        f_s[i] = f
    
    F_s = np.absolute(f_s)**2
    F_s = np.transpose(F_s) #so that x_0 is horizontal, k is vertical
    return F_s

def generate_phase_distribution(psi_s,x):
    F_s = []
    for psi in psi_s:
        F = phase_distribution(psi,x)
        F_s.append(F)
    return F_s


def generate_phase_plots(psi_s,x,dx,hbar,y_max,max_F,frame_spacing, Directory,folder_name):
    
    k = 2*np.pi*np.fft.fftfreq(len(x),dx)
    x_min, x_max = np.min(x), np.max(x)
    k_min, k_max = np.min(k), np.max(k)
    
    #directory = "C:\\Users\\boris\\OneDrive - Queen's University\\Documents\\JOB_Files\\McDonald Institute Fellowship\\Research\\Coding"
    path = Directory+"\\"+folder_name

    #y_max = 10

    for i in range(0,len(psi_s),frame_spacing):
        F = Husimi_phase(psi_s[i],x,dx,hbar)
        #fig,ax = plt.subplots(1,1)
        #plt.figure(figsize = (10,5))
        plt.imshow(F,extent = (x_min,x_max,k_min,k_max),cmap = cm.hot, norm = Normalize(0,max_F), aspect = (x_max-x_min)/(2*y_max))
        plt.xlim([x_min,x_max])
        plt.ylim([-y_max,y_max])
        plt.colorbar()
        
        #now save it as a .jpg file:
        filename = 'ToyModelPlot' + str(i+1).zfill(4) + '.jpg'
        folder = path #"C:\\Users\\boris\\OneDrive - Queen's University\\Documents\\JOB_Files\\McDonald Institute Fellowship\\Research\\Coding\\"+folder_name
        plt.savefig(folder + "\\" + filename)  #save this figure (includes both subplots)
        
        plt.close() #close plot so it doesn't overlap with the next one
        print(i,' Done')


# Video Generating function
def generate_phase_video(Directory, folder_name, video_name, dt, frame_spacing):
    os.chdir(Directory)
    image_folder  = Directory +"/"+folder_name
    images = [img for img in os.listdir(image_folder)
            if img.endswith(".jpg") or
                img.endswith(".jpeg") or
                img.endswith("png")]
    
    # Array images should only consider
    # the image files ignoring others if any
    print(images) 

    frame = cv2.imread(os.path.join(image_folder, images[0]))

    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape  

    fps = int(1/(dt*frame_spacing))
    video = cv2.VideoWriter(video_name, 0, fps, (width, height)) 

    # Appending the images to the video one by one
    for image in images: 
        video.write(cv2.imread(os.path.join(image_folder, image))) 
    
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    video.release()  # releasing the video generated


def animate_phase(Directory, folder_name: str, video_name: str, dt, frame_spacing):
    path = Directory+"\\"+folder_name

    # Folder which contains all the images
    # from which video is to be generated
    os.chdir(path)  
    
    mean_height = 0
    mean_width = 0
    
    num_of_images = len(os.listdir('.'))
    #print(num_of_images)
    
    for file in os.listdir('.'):
        im = Image.open(os.path.join(path, file))
        width, height = im.size
        mean_width += width
        mean_height += height
        # im.show()   # uncomment this for displaying the image
    
    # Finding the mean height and width of all images.
    # This is required because the video frame needs
    # to be set with same width and height. Otherwise
    # images not equal to that width height will not get 
    # embedded into the video
    mean_width = int(mean_width / num_of_images)
    mean_height = int(mean_height / num_of_images)
    
    # print(mean_height)
    # print(mean_width)
    
    # Resizing of the images to give
    # them same width and height 
    for file in os.listdir('.'):
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"):
            # opening image using PIL Image
            im = Image.open(os.path.join(path, file)) 
    
            # im.size includes the height and width of image
            width, height = im.size   
            #print(width, height)
    
            # resizing 
            imResize = im.resize((mean_width, mean_height), Image.ANTIALIAS) 
            imResize.save( file, 'JPEG', quality = 1080) # setting quality
            # printing each resized image name
            #print(im.filename.split('\\')[-1], " is resized") 
    
    # Calling the generate_video function
    generate_phase_video()
