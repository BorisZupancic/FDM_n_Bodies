pwd
cd ..
cd ..
cd ..
cd ..
cd ..
pwd

parent="Analysis/TestingEquilibriumICs/Full"

#Possible choices for number of Particles:
#num_particles=('5' '500' '1000' '5000' '10000' '50000' '100000')

#Boundary Conditions:
bc_choice='1'

#Choices of either full simulation video or snapshots:
simulation_choice2=('2')

#Number of Collapse times to run for:
dynamical_times='300'

#Initial Conditions:
IC_choice='3'

#Variable Mass:
var_mass='n'

#Mass ratios:
percent='0.5'

#Number of Particles:
num_p='50000'

v_rms=0.28
z_rms=0.07
lambdas=('0.05')  #('1' '0.5' '0.1' '0.05')

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
Variable Mass : %s \n \n' "2" "$percent" "$v_rms" "$z_rms" "$lambda" "$num_p" "n" "$bc_choice" "$sim" "$dynamical_times" "$IC_choice" "$var_mass"

    printf '%s\n%f\n%f\n%f\n%f\n%f\n%i\n%s\n%i\n%i\n%i\n%i\n%s' "$parent"  "2" "$percent" "$v_rms" "$z_rms" "$lambda" "$num_p" "n" "$bc_choice" "$sim" "$dynamical_times" "$IC_choice" "$var_mass" | python3 -u 1D_Codes/Programs/ProgramV2.py
done

done
