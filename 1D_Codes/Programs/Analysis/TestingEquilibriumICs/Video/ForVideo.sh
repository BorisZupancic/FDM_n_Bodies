pwd
cd ..
cd ..
cd ..
cd ..
cd ..
pwd

parent="Analysis/TestingEquilibriumICs/Video"

#Possible choices for number of Particles:
#num_particles=('5' '500' '1000' '5000' '10000' '50000' '100000')
#Choices of either full simulation video or snapshots:
simulation_choice2=('1') 

#Boundary Conditions:
bc_choice='1'

collapse_times='5'

IC=3

#Mass ratios:
percent='0.5'

#Number of Particles:
num_p='10000'

#variable mass:
var_mass='n'

#Run through just particles:
v_rms=0.58
z_rms=0.158
lambdas=0.05 # '1' '0.5' '0.1')

for lambda in "${lambdas[@]}"; do

for sim in "${simulation_choice2[@]}"; do
    echo "---------------------New Sim---------------------"
    printf 'L = %f
FDM percentage by mass : %f 
v_FDM dispersion = %f
Characteristic size: R_syst = %f
de Broglie Wavelength = %f
Num_particles = %i
Fixed Phi: %s
BC choice: %i
sim choice : %i
collapse times : %i
IC choice : %i 
Variable Mass : %s \n \n' "2" "$percent" "$v_rms" "$z_rms" "$lambda" "$num_p" "n" "$bc_choice" "$sim" "$collapse_times" "$IC" "$var_mass"

    printf '%s\n%f\n%f\n%f\n%f\n%f\n%i\n%s\n%i\n%i\n%i\n%i\n%s' "$parent"  "2" "$percent" "$v_rms" "$z_rms" "$lambda" "$num_p" "n" "$bc_choice" "$sim" "$collapse_times" "$IC" "$var_mass" | python3 -u 1D_Codes/Programs/ProgramV2.py
done

done
