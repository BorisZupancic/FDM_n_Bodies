pwd
cd ..
cd ..
cd ..
cd ..
pwd

parent="Analysis/TestingFDM"
#Possible choices for deB wavelength:
de_Broglie=('2' '1' '0.5' '0.1' '0.05' '0.01')
#Choice of Fixed potential:
fixedPhi='n'
#Choices of either full simulation video or snapshots:
simulation_choice2=('2') #want it to run a long time

#Mass ratios:
percent='1'
v_rms=0.57551038
z_rms=0.15924094
    
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
sim choice : %i
Boson std = %f
Particle std = %f \n \n' "2" "$percent" "$v_rms" "$z_rms" "$lambda" "$fixedPhi" "$sim" "0.1"

        printf '%s\n%f\n%f\n%f\n%f\n%f\n%s\n%i\n%f\n%f' "$parent" "2" "$percent" "$v_rms" "$z_rms" "$lambda" "$fixedPhi" "$sim" "0.1" | python3 -u 1D_Codes/Programs/ProgramV2.py 
    done
done
