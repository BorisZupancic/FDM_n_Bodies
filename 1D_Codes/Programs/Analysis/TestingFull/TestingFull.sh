pwd
cd ..
cd ..
cd ..
cd ..
pwd

parent="Analysis/TestingFull"

#Possible choices for number of Particles:
#num_particles=('5' '500' '1000' '5000' '10000' '50000' '100000')
#Choices of either full simulation video or snapshots:
simulation_choice2=('2') #want it to run a long time

#Mass ratios:
percent='0.5'

#Number of Particles:
num_p='100000'

#Run through just particles:
v_rms=0.58 #0.22
z_rms=0.158 #0.17
lambdas=('1' '0.5' '0.1' '0.05' '0.01') # '1'

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
sim choice : %i
Boson std = %f
Particle std = %f \n \n' "2" "$percent" "$v_rms" "$z_rms" "$lambda" "$num_p" "n" "$sim" "0.1" "0.1"

    printf '%s\n%f\n%f\n%f\n%f\n%f\n%i\n%s\n%i\n%f\n%f' "$parent"  "2" "$percent" "$v_rms" "$z_rms" "$lambda" "$num_p" "n" "$sim" "0.1" "0.1" | python3 -u 1D_Codes/Programs/ProgramV2.py
done

done