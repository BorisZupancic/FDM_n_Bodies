pwd
cd ..
cd ..
cd ..
cd ..
cd ..
pwd

parent="Analysis/TestingEquilibriumICs/FDM"
#Possible choices for deB wavelength:
de_Broglie='0.1' #('2' '1' '0.5' '0.1' '0.05' '0.01')
#Choice of Fixed potential:
fixed_phi='n'

#Boundary Conditions:
bc_choice='1'

#Choices of either full simulation video or snapshots:
simulation_choice2=('2') #want it to run a long time

#Number of Collapse times to run for:
collapse_times='50'

#Initial Conditions:
IC_choice='3'

#Variable Mass:
var_mass='n'

#Percentage of FDM by mass:
percent='1'

v_rms=0.28
z_rms=0.07
    
#Run through just particles:
for lambda in "${de_Broglie[@]}"; do
    
    for sim in "${simulation_choice2[@]}"; do
        echo "---------------------New Sim---------------------"
        printf 'L = %f
FDM percentage by mass : %f 
v_FDM dispersion = %f
Characteristic Radius: R_syst = %f
de Broglie Wavelength = %f
Fixed Potential: %s
BC choice : %i
sim choice : %i
Number collapse times: %i 
ICs choice : %i
Variable Mass : %s \n \n' "2" "$percent" "$v_rms" "$z_rms" "$lambda" "$fixed_phi" "$bc_choice" "$sim" "$collapse_times" "$IC_choice" "$var_mass"

        printf '%s\n%f\n%f\n%f\n%f\n%f\n%s\n%i\n%i\n%i\n%i\n%s' "$parent" "2" "$percent" "$v_rms" "$z_rms" "$lambda" "$fixed_phi" "$bc_choice" "$sim" "$collapse_times" "$IC_choice" "$var_mass" | python3 -u 1D_Codes/Programs/ProgramV2.py 
    done
done
