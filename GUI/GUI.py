import tkinter as tk
import subprocess
import os

cwd = os.getcwd()
print(cwd)
GUI_path = cwd #+ "/1D_Codes/GUI"
print(GUI_path)

master = tk.Tk()

#Setup Frames
height = 200
width = 300

key0,key1,key2 = "Initial Setup","Particles", "FDM"
Initial = tk.LabelFrame(master, text=key0, bd = 5, padx = 30, pady=50)
Particles = tk.LabelFrame(master, text=key1, bd = 5, padx = 30, pady=50, height = height, width = width)
FDM = tk.LabelFrame(master, text=key2, bd = 5, padx = 30, pady=50, height = height, width = width)


L = tk.Entry(Initial)
percent_FDM = tk.Entry(Initial)
fixed_phi = tk.Entry(Initial)
sim_choice = tk.Entry(Initial)

Num_stars = tk.Entry(Particles)
particle_std = tk.Entry(Particles)

v_FDM = tk.Entry(FDM)
R = tk.Entry(FDM) 
lambda_deB = tk.Entry(FDM)
FDM_std = tk.Entry(FDM)


values0,values1,values2 = [ 
                            {"Box Length": L, "Percent of FDM": percent_FDM, "Fixed Potential? [y/n]:":fixed_phi,"Full Video or Snapshots? [1/2]": sim_choice},
                            {"Number of Particles": Num_stars, "Initial Standard Deviation (as fraction of the box width)": particle_std},
                            {"FDM Velocity dispersion v_FDM": v_FDM,"Characteristic Size R": R,"DeBroglie Wavelength (Ratio to R)": lambda_deB,"Initial Standard Deviation (as fraction of the box width)": FDM_std}
                            ]

Fields = {key0: values0, 
key1: values1, 
key2: values2}



#Initial:
Initial.grid(row=0, column= 0,columnspan=4)
row_i = 1
for key, val in Fields[key0].items():
    tk.Label(Initial, text=key).grid(row=row_i, column= 0,columnspan=2)
    val = tk.Entry(Initial)
    #print(val)
    val.grid(row=row_i,column=2,columnspan=2)
    row_i+=1


#Particles
Particles.grid(row=row_i, column= 0,columnspan=2)
row1_i = row_i+1
for key,val in Fields[key1].items():
    tk.Label(Particles, text=key).grid(row=row1_i, column=0,columnspan=1)
    val = tk.Entry(Particles)
    val.grid(row=row1_i,column=1,columnspan=1)
    row1_i+=1

#FDM
FDM.grid(row=row_i, column= 2,columnspan=2)
row2_i = row_i+1
for key,val in Fields[key2].items():
    tk.Label(FDM, text=key).grid(row=row2_i, column=2,columnspan=1)
    val = tk.Entry(FDM)
    #print(key,val)
    
    val.grid(row=row2_i,column=3,columnspan=1)
    row2_i+=1

#print(Fields)



def get_inputs():

    #redefine what args are, by accessing dictionary
    #...they were NoneType before
    values0 = Fields[key0]
    values1 = Fields[key1]
    values2 = Fields[key2]
    print(values0,values1,values2)

    L = values0["Box Length"]
    percent_FDM = values0["Percent of FDM"]
    fixed_phi = values0["Fixed Potential? [y/n]:"]
    sim_choice = values0["Full Video or Snapshots? [1/2]"]

    Num_stars = values1["Number of Particles"]
    particle_std = values1["Initial Standard Deviation (as fraction of the box width)"]

    v_FDM = values2["FDM Velocity dispersion v_FDM"]
    R = values2["Characteristic Size R"]
    lambda_deB = values2["DeBroglie Wavelength (Ratio to R)"]
    FDM_std = values2["Initial Standard Deviation (as fraction of the box width)"]
    
    #print(L,values0["Box Length"])

    # if L is not None:             
    #     args = [L.get(),
    #         percent_FDM.get(),
    #         v_FDM.get(),
    #         R.get(),
    #         lambda_deB.get(),
    #         Num_stars.get(),
    #         fixed_phi.get(),
    #         sim_choice.get(),
    #         particle_std.get(),
    #         FDM_std.get()
    #         ]
    #    return args

    args = [L.get(),
            percent_FDM.get(),
            v_FDM.get(),
            R.get(),
            lambda_deB.get(),
            Num_stars.get(),
            fixed_phi.get(),
            sim_choice.get(),
            particle_std.get(),
            FDM_std.get()
            ]
    return args

def run_Program():
    args = get_inputs()
    print("args:",args)
    #os.chdir(GUI_path)
    #print(os.getcwd())
    #print(os.listdir())
    subprocess.check_call("callProgram.sh",args,shell=True)
    #os.chdir(cwd)

Run = tk.Button(master,text = "Run",command = run_Program)
Run.grid(row = row_i +1, column=0)

master.mainloop()