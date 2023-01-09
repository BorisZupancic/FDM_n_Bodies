pwd
cd ..
cd ..
cd ..
cd ..
cd ..
pwd

parent="Analysis/TestingIsolated/FDM"
#Possible choices for deB wavelength:
de_Broglie=('2' '1' '0.5' '0.1' '0.05') #'0.01')
#Choice of Fixed potential:
fixedPhi='n'
#Choices of either full simulation video or snapshots:
simulation_choice2=('2') #want it to run a long time
bc_choice='1'


#Number of Collapse times to run for:
collapse_times='20'
#Initial Conditions:
IC_choice='1'


#Mass ratios:
percent='1'
v_rms=0.58
z_rms=0.158
    
#Run through just particles:
for lambda in "${de_Broglie[@]}"; do
    
    for sim in "${simulation_choice2[@]}"; do
        echo "---------------------New Sim---------------------"
        printf 'L = %f
FDM percentage by mass : %f 
v_FDM dispersion = %f
Characteristic Radius: R_syst = %f
de Broglie Wavelength = %f
Fixed Phi? %s
BC choice : %i
sim choice : %i
Number collapse times: %i 
ICs choice : %i
Boson std = %f
Particle std = %f \n \n' "2" "$percent" "$v_rms" "$z_rms" "$lambda" "$fixedPhi" "$bc_choice" "$sim" "$collapse_times" "$IC_choice" "0.1"

        printf '%s\n%f\n%f\n%f\n%f\n%f\n%s\n%i\n%i\n%i\n%i\n%f\n%f' "$parent" "2" "$percent" "$v_rms" "$z_rms" "$lambda" "$fixedPhi" "$bc_choice" "$sim" "$collapse_times" "$IC_choice" "0.1" | python3 -u 1D_Codes/Programs/ProgramV2.py 
    done
done
